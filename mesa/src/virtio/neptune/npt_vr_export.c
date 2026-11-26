/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * VR eye-texture export for steamac's OpenVR runtime under Obelisk.
 *
 * Proton's vrclient_x64, as patched by steamac (guest/proton-vrclient), hands a D3D11 eye texture to the OpenVR
 * runtime only through DXMT's IMTLD3D11ExportableTexture: GetMetalTexture returns three opaque 64-bit values that
 * reach the runtime untouched (svr_metal_texture.handle/event_handle/event_value). Neptune answers that interface
 * here, the way its own swapchain feeds the wsd window server:
 *
 *   - a pool of exportable host textures (MISC_SHARED | LINEAR_EXPORT) per eye-texture shape, each bound once to a
 *     guest blob whose fd any macOS process can map (npt_renderer_create_host_blob);
 *   - GetMetalTexture copies every array slice of the eye texture into the next pool image, stacked vertically,
 *     flushes, and waits for the copy on an EVENT query;
 *   - it returns handle = NPT_VR_HANDLE_MAGIC << 56 | pitch << 24 | fd. The runtime (same process) sends that fd by
 *     SCM_RIGHTS to obelisk-vrbridge, which maps it as a linear MTLTexture and gives it to steamac-vrhost.
 *
 * Prototype: pools are never freed, and the wait is a bounded poll of the EVENT query.
 */
#include "npt_vr_export.h"

#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "npt_com.h"
#include "npt_device.h"
#include "npt_event.h"
#include "npt_ring.h"
#include "npt_swapchain_common.h"
#include "npt_renderer.h"
#include "npt_shared_texture.h"
#include "neptune-protocol/npt_protocol_client_id3d11device.h"
#include "neptune-protocol/npt_protocol_client_id3d11devicecontext.h"
#include "neptune-protocol/npt_protocol_client_id3d11texture2d.h"
#include "neptune-protocol/npt_protocol_client_id3d11fence.h"
#include "neptune-protocol/npt_protocol_guest_id3d11devicecontext.h"
#include "neptune-protocol/npt_protocol_guest_id3d11fence.h"
#include "neptune-protocol/npt_protocol_defs.h"
#include "neptune-protocol/npt_protocol_directx_types.h"

/* ad6245d1-3a11-4dd6-87b7-7a58048b55d3: DXMT's IMTLD3D11ExportableTexture (steamac's dxmt-interop.h) */
static const GUID npt_iid_exportable_texture = {
   0xad6245d1, 0x3a11, 0x4dd6, {0x87, 0xb7, 0x7a, 0x58, 0x04, 0x8b, 0x55, 0xd3}};
static const GUID npt_iid_unknown = {
   0x00000000, 0x0000, 0x0000, {0xc0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x46}};

#define NPT_VR_IMAGES 6 /* headroom: the bridge shows an image only once its copy's fence signals */
#define NPT_VR_HANDLE_MAGIC 0xA5ull
#define NPT_VR_FENCE_MAGIC 0xA6ull /* shared_event = magic << 56 | fd that signals once the copy is done */
#define NPT_VR_WAIT_MAX_MS 20
#define NPT_VR_MAX_POOLS 4
#define NPT_VR_SAME_FRAME_MS 8

struct npt_vr_pool {
   struct npt_vr_pool *next;
   void *device;                          /* ID3D11Device wrapper (its ref is the pool's) */
   uint64_t device_id;                    /* its object id: GetDevice may hand out another wrapper each call */
   uint32_t width, height, slices, format; /* per slice; images are width x (height * slices) */
   void *image[NPT_VR_IMAGES];            /* ID3D11Texture2D wrappers */
   int fd[NPT_VR_IMAGES];
   uint32_t pitch[NPT_VR_IMAGES];
   uint32_t next_image;
   void *query;                           /* ID3D11Query, D3D11_QUERY_EVENT: the fallback wait */
   /* GPU-done fence, as Neptune's swapchain does it (gsc_present_common): Signal after the copy, a token armed on
    * its value turns into a sync fd the bridge waits on, so the game never waits for the GPU here */
   void *ctx;                             /* the immediate context, kept: its ring and ids stay valid */
   void *ctx4;                            /* ID3D11DeviceContext4 */
   uint64_t ctx4_id;
   void *fence;                           /* ID3D11Fence */
   uint64_t fence_id, fence_value;
   uint64_t token[NPT_VR_IMAGES];         /* armed for each image's last copy; released when the image is reused */
   uint64_t value_of[NPT_VR_IMAGES];      /* fence value of each image's last copy */
   uint32_t arm_failures;
};

/* Neptune's mtx_t is a CRITICAL_SECTION without a static initializer; an SRW lock has one */
static SRWLOCK g_lock = SRWLOCK_INIT;
static struct npt_vr_pool *g_pools;
/* the last copy, for the other eye of the same frame (under g_lock) */
static struct {
   uint64_t tex_id, handle, value;
   int64_t when; /* QueryPerformanceCounter at the copy */
} g_last;
/* copy waits, for the log (under g_lock) */
static struct {
   uint32_t frames, timeouts;
   uint64_t polls;
   double wait_ms, max_ms;
} g_stats;

static bool
iid_equal(const GUID *a, const GUID *b)
{
   return !memcmp(a, b, sizeof(*a));
}

static const struct npt_id3d11device_client_vtbl *
dev_vtbl(void *device)
{
   return (const void *)((struct npt_com_base *)device)->lpVtbl;
}
static const struct npt_id3d11devicecontext_client_vtbl *
ctx_vtbl(void *ctx)
{
   return (const void *)((struct npt_com_base *)ctx)->lpVtbl;
}
static const struct npt_id3d11texture2d_client_vtbl *
tex_vtbl(void *tex)
{
   return (const void *)((struct npt_com_base *)tex)->lpVtbl;
}

/* The pool for this shape, made on first use. Under g_lock. */
static struct npt_vr_pool *
pool_get(void *device, uint32_t width, uint32_t height, uint32_t slices, uint32_t format)
{
   const uint64_t device_id = npt_com_self_id(device); /* for the log only */
   uint32_t pools = 0;
   for (struct npt_vr_pool *p = g_pools; p; p = p->next, pools++) {
      /* by shape only: every GetDevice hands out a fresh wrapper with its own id, and a game renders its
       * eye textures on one device */
      if (p->width == width && p->height == height && p->slices == slices && p->format == format)
         return p;
   }
   /* pools are never freed: a game that keeps changing its eye texture shape must not eat the GPU's memory */
   if (pools >= NPT_VR_MAX_POOLS) {
      static bool warned;
      if (!warned)
         npt_log("vr export: %u pools already, refusing %ux%u x%u fmt %u", pools, width, height, slices, format);
      warned = true;
      return NULL;
   }
   struct npt_vr_pool *p = calloc(1, sizeof(*p));
   if (!p)
      return NULL;
   p->device = device;
   p->device_id = device_id;
   p->width = width, p->height = height, p->slices = slices, p->format = format;
   for (uint32_t i = 0; i < NPT_VR_IMAGES; i++)
      p->fd[i] = -1;

   for (uint32_t i = 0; i < NPT_VR_IMAGES; i++) {
      p->image[i] = npt_shared_texture_create_exportable(device, width, height * slices, format);
      if (!p->image[i])
         goto fail;
      struct npt_shared_texture_desc exp;
      memset(&exp, 0, sizeof(exp));
      if (!npt_shared_texture_export_blob(p->image[i], &exp))
         goto fail;
      if (exp.plane_count != 1 || exp.modifier != 0 || exp.planes[0].offset != 0)
         npt_log("vr export: unexpected export (planes=%u modifier=0x%llx offset=%llu)", exp.plane_count,
                 (unsigned long long)exp.modifier, (unsigned long long)exp.planes[0].offset);
      struct npt_device *dev = npt_com_self_device(p->image[i]);
      const uint64_t size = (exp.allocation_size + 4095u) & ~4095ull;
      p->fd[i] = npt_renderer_create_host_blob(dev->renderer, exp.blob_id, size);
      p->pitch[i] = (uint32_t)exp.planes[0].pitch;
      if (p->fd[i] < 0)
         goto fail;
   }
   dev_vtbl(device)->GetImmediateContext(device, (ID3D11DeviceContext **)&p->ctx);
   if (p->ctx &&
       NPT_SUCCEEDED(ctx_vtbl(p->ctx)->QueryInterface(p->ctx, &NPT_IID_ID3D11DeviceContext4, &p->ctx4)) && p->ctx4) {
      p->ctx4_id = npt_com_self_id(p->ctx4);
      void *device5 = NULL;
      if (NPT_SUCCEEDED(dev_vtbl(device)->QueryInterface(device, &NPT_IID_ID3D11Device5, &device5)) && device5) {
         const struct npt_id3d11device5_client_vtbl *v5 = (const void *)((struct npt_com_base *)device5)->lpVtbl;
         if (NPT_FAILED(v5->CreateFence(device5, 0, D3D11_FENCE_FLAG_NONE, &NPT_IID_ID3D11Fence, &p->fence)))
            p->fence = NULL;
         v5->Release(device5);
      }
      if (p->fence)
         p->fence_id = npt_com_self_id(p->fence);
   }
   D3D11_QUERY_DESC qd;
   memset(&qd, 0, sizeof(qd));
   qd.Query = D3D11_QUERY_EVENT;
   if (NPT_FAILED(dev_vtbl(device)->CreateQuery(device, &qd, (ID3D11Query **)&p->query)))
      p->query = NULL;
   npt_log("vr export: pool %ux%u x%u slices fmt %u: fds %d %d %d pitch %u, %s", width, height, slices, format,
           p->fd[0], p->fd[1], p->fd[2], p->pitch[0],
           p->fence ? "GPU fence (the bridge waits)" : p->query ? "event query (the game waits)" : "unsynchronized");
   p->next = g_pools;
   g_pools = p;
   return p;

fail:
   npt_log("vr export: pool %ux%u x%u fmt %u failed", width, height, slices, format);
   /* leaked on purpose (prototype): partial pools are rare and small */
   return NULL;
}

struct npt_vr_export;
struct npt_vr_export_vtbl {
   HRESULT(NPT_STDMETHODCALLTYPE *QueryInterface)(struct npt_vr_export *self, REFIID riid, void **object);
   ULONG(NPT_STDMETHODCALLTYPE *AddRef)(struct npt_vr_export *self);
   ULONG(NPT_STDMETHODCALLTYPE *Release)(struct npt_vr_export *self);
   HRESULT(NPT_STDMETHODCALLTYPE *GetMetalTexture)(struct npt_vr_export *self, UINT64 *texture, UINT64 *shared_event,
                                                   UINT64 *value);
};
struct npt_vr_export {
   const struct npt_vr_export_vtbl *lpVtbl;
   atomic_uint ref;
   void *texture; /* ID3D11Texture2D wrapper (its ref is ours) */
};

static ULONG NPT_STDMETHODCALLTYPE
vx_AddRef(struct npt_vr_export *self)
{
   return atomic_fetch_add(&self->ref, 1) + 1;
}

static ULONG NPT_STDMETHODCALLTYPE
vx_Release(struct npt_vr_export *self)
{
   ULONG r = atomic_fetch_sub(&self->ref, 1) - 1;
   if (!r) {
      tex_vtbl(self->texture)->Release(self->texture);
      free(self);
   }
   return r;
}

static HRESULT NPT_STDMETHODCALLTYPE
vx_QueryInterface(struct npt_vr_export *self, REFIID riid, void **object)
{
   if (!object)
      return NPT_E_POINTER;
   if (iid_equal(riid, &npt_iid_exportable_texture) || iid_equal(riid, &npt_iid_unknown)) {
      vx_AddRef(self);
      *object = self;
      return NPT_S_OK;
   }
   return tex_vtbl(self->texture)->QueryInterface(self->texture, riid, object);
}

static HRESULT NPT_STDMETHODCALLTYPE
vx_GetMetalTexture(struct npt_vr_export *self, UINT64 *texture, UINT64 *shared_event, UINT64 *value)
{
   void *tex = self->texture;
   /* Both eyes of one frame in one texture (side by side, or an array): OpenVR submits it once per eye, back to
    * back. The second submit within a few ms of the first, of the same texture, reuses the first copy. */
   const uint64_t tex_id = npt_com_self_id(tex);
   LARGE_INTEGER now, freq;
   QueryPerformanceCounter(&now);
   QueryPerformanceFrequency(&freq);
   AcquireSRWLockExclusive(&g_lock);
   if (g_last.handle && g_last.tex_id == tex_id &&
       (uint64_t)(now.QuadPart - g_last.when) * 1000 < (uint64_t)freq.QuadPart * NPT_VR_SAME_FRAME_MS) {
      *texture = g_last.handle;
      *shared_event = 0;
      *value = g_last.value;
      g_last.handle = 0; /* one reuse per copy: a third submit is a new frame */
      ReleaseSRWLockExclusive(&g_lock);
      return NPT_S_OK;
   }
   ReleaseSRWLockExclusive(&g_lock);

   D3D11_TEXTURE2D_DESC d;
   memset(&d, 0, sizeof(d));
   tex_vtbl(tex)->GetDesc(tex, &d);
   if (d.SampleDesc.Count > 1 || !d.Width || !d.Height)
      return NPT_E_FAIL;
   const uint32_t slices = d.ArraySize ? d.ArraySize : 1;
   const uint32_t format = npt_shared_texture_host_format(d.Format);

   ID3D11Device *device = NULL;
   tex_vtbl(tex)->GetDevice(tex, &device);
   if (!device)
      return NPT_E_FAIL;

   AcquireSRWLockExclusive(&g_lock);
   struct npt_vr_pool *p = pool_get(device, d.Width, d.Height, slices, format);
   if (!p) {
      ReleaseSRWLockExclusive(&g_lock);
      dev_vtbl(device)->Release(device);
      return NPT_E_FAIL;
   }
   if (p->device != device)
      dev_vtbl(device)->Release(device); /* an existing pool holds this device already */
   const uint32_t i = p->next_image;
   p->next_image = (i + 1) % NPT_VR_IMAGES;

   void *ctx = p->ctx;
   if (!ctx) {
      ReleaseSRWLockExclusive(&g_lock);
      return NPT_E_FAIL;
   }
   const struct npt_id3d11devicecontext_client_vtbl *cv = ctx_vtbl(ctx);
   /* Backpressure: the image is free once its previous copy has run on the GPU (the bridge shows it right after;
    * NPT_VR_IMAGES - 1 frames later it is long done). Without it a GPU-bound game queues frames ahead of the GPU,
    * and a copy can overwrite an image the bridge is about to show with the pose of an older frame. */
   if (p->fence && p->value_of[i]) {
      const struct npt_id3d11fence_client_vtbl *fv = (const void *)((struct npt_com_base *)p->fence)->lpVtbl;
      LARGE_INTEGER b0, b1, bf;
      QueryPerformanceFrequency(&bf);
      QueryPerformanceCounter(&b0);
      while (fv->GetCompletedValue(p->fence) < p->value_of[i]) {
         QueryPerformanceCounter(&b1);
         if ((b1.QuadPart - b0.QuadPart) * 1000 > bf.QuadPart * NPT_VR_WAIT_MAX_MS * 5)
            break; /* 100 ms: a stalled GPU must not freeze the game for good */
         Sleep(0);
      }
   }
   const uint32_t mips = d.MipLevels ? d.MipLevels : 1;
   for (uint32_t s = 0; s < slices; s++)
      cv->CopySubresourceRegion(ctx, (ID3D11Resource *)p->image[i], 0, 0, s * d.Height, 0, (ID3D11Resource *)tex,
                                s * mips, NULL);
   /* fenced: Signal after the copy, Flush (submits both), arm a token on the value -> sync fd for the bridge */
   uint64_t fence_handle = 0, fence_value = 0;
   if (p->fence) {
      fence_value = ++p->fence_value;
      npt_async_ID3D11DeviceContext4_Signal(npt_com_self_ring(ctx), p->ctx4_id, (ID3D11Fence *)p->fence, fence_value);
      cv->Flush(ctx);
      struct npt_device *ndev = npt_com_self_device(ctx);
      const uint64_t token = npt_gsc_make_token();
      const int ffd = npt_event_arm_token_fd(ndev, token);
      if (ffd < 0 && (++p->arm_failures <= 5 || p->arm_failures % 100 == 0))
         npt_log("vr export: fence arm failed (%u so far): this frame waits on the event query", p->arm_failures);
      if (ffd >= 0) {
         npt_async_ID3D11Fence_SetEventOnCompletion(npt_device_method_ring(ndev), p->fence_id, fence_value,
                                                    (HANDLE)(uintptr_t)token);
         /* this image's previous copy is 3 frames old: its wait is long over */
         if (p->token[i])
            npt_event_release_token(ndev, p->token[i]);
         p->token[i] = token;
         p->value_of[i] = fence_value;
         fence_handle = NPT_VR_FENCE_MAGIC << 56 | (uint64_t)(ffd & 0xffffff);
      }
   }
   if (!fence_handle && p->query)
      cv->End(ctx, (ID3D11Asynchronous *)p->query);
   if (!fence_handle)
      cv->Flush(ctx);
   /* wait for the copy, bounded in time: a stalled host must not stall the game's render thread for seconds */
   LARGE_INTEGER w0, w1, qf;
   QueryPerformanceFrequency(&qf);
   QueryPerformanceCounter(&w0);
   bool done = fence_handle || !p->query;
   uint32_t polls = 0;
   const int64_t limit = qf.QuadPart * NPT_VR_WAIT_MAX_MS / 1000;
   for (; !done; polls++) {
      BOOL finished = FALSE;
      HRESULT hr = cv->GetData(ctx, (ID3D11Asynchronous *)p->query, &finished, sizeof(finished), 0);
      done = hr == NPT_S_OK && finished;
      QueryPerformanceCounter(&w1);
      if (!done && w1.QuadPart - w0.QuadPart > limit)
         break;
   }
   QueryPerformanceCounter(&w1);
   const double wait_ms = (double)(w1.QuadPart - w0.QuadPart) * 1000.0 / (double)qf.QuadPart;
   g_stats.frames++, g_stats.wait_ms += wait_ms, g_stats.polls += polls;
   if (wait_ms > g_stats.max_ms) g_stats.max_ms = wait_ms;
   if (!done) {
      g_stats.timeouts++;
      if (g_stats.timeouts <= 5 || g_stats.timeouts % 100 == 0)
         npt_log("vr export: copy not done after %.1f ms (%u polls): %u timeouts so far", wait_ms, polls,
                 g_stats.timeouts);
   }
   if (g_stats.frames % 300 == 0) {
      npt_log("vr export: %u copies, wait avg %.2f ms max %.2f ms, %.1f polls avg, %u timeouts", g_stats.frames,
              g_stats.wait_ms / 300.0, g_stats.max_ms, (double)g_stats.polls / 300.0, g_stats.timeouts);
      g_stats.wait_ms = 0, g_stats.max_ms = 0, g_stats.polls = 0;
   }
   const int fd = p->fd[i];
   const uint32_t pitch = p->pitch[i];
   ReleaseSRWLockExclusive(&g_lock);

   *texture = NPT_VR_HANDLE_MAGIC << 56 | ((uint64_t)(pitch & 0xffffffu) << 24) | (uint64_t)(fd & 0xffffff);
   *shared_event = fence_handle;
   *value = fence_handle ? fence_value : i;
   LARGE_INTEGER done_at;
   QueryPerformanceCounter(&done_at);
   AcquireSRWLockExclusive(&g_lock);
   g_last.tex_id = tex_id, g_last.handle = *texture, g_last.value = i, g_last.when = done_at.QuadPart;
   ReleaseSRWLockExclusive(&g_lock);
   return NPT_S_OK;
}

static const struct npt_vr_export_vtbl npt_vr_export_vtbl = {
   vx_QueryInterface,
   vx_AddRef,
   vx_Release,
   vx_GetMetalTexture,
};

bool
npt_vr_export_query(void *self, const GUID *riid, void **ppvObject)
{
   if (!riid || !iid_equal(riid, &npt_iid_exportable_texture))
      return false;
   /* only 2D textures: ask the object itself (its generated QueryInterface knows its own chain) */
   void *tex2d = NULL;
   const struct npt_id3d11texture2d_client_vtbl *v = (const void *)((struct npt_com_base *)self)->lpVtbl;
   if (NPT_FAILED(v->QueryInterface(self, &NPT_IID_ID3D11Texture2D, &tex2d)) || !tex2d)
      return false;
   struct npt_vr_export *x = calloc(1, sizeof(*x));
   if (!x) {
      tex_vtbl(tex2d)->Release(tex2d);
      return false;
   }
   x->lpVtbl = &npt_vr_export_vtbl;
   atomic_init(&x->ref, 1);
   x->texture = tex2d;
   *ppvObject = x;
   return true;
}
