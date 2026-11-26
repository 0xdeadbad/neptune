/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * VR eye-texture export for steamac's OpenVR runtime under Obelisk (obelisk-vrbridge).
 */
#ifndef NPT_VR_EXPORT_H
#define NPT_VR_EXPORT_H

#include <stdbool.h>

#include "npt_common.h"

/* QueryInterface hook: answers IID_IMTLD3D11ExportableTexture on an ID3D11Texture2D with Neptune's export object.
 * Returns false (nothing done) for any other IID or object. */
bool npt_vr_export_query(void *self, const GUID *riid, void **ppvObject);

#endif
