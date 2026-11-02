# Offline Scene Environment Asset

## Provenance and License

`ui/public/hdr/studio_small_03_1k.hdr` is Poly Haven's Studio Small 03 HDRI by
Greg Zaal, released under CC0. The [official asset page](https://polyhaven.com/a/studio_small_03)
identifies its creator and license.

- Source file: [1K HDR download](https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/1k/studio_small_03_1k.hdr)
- Verified size: 1,686,299 bytes
- Poly Haven API MD5: `74e6ef69ea9024c2cc25b3a7de8ec2f7`
- SHA-256: `30933d55e45f0795daf49f3cbefbe0e5ebcb821ee04fb0a2818c02ffc3938817`

## Runtime and Packaging

`SceneEnvironment` requests the asset at `/hdr/studio_small_03_1k.hdr` from the
local application origin. Vite copies files in `ui/public` to `ui/dist`; Tauri's
`frontendDist` points to `../dist`.

The production build could not verify that copy in this environment. Both
`npm run build` and a direct Vite build stop at the unrelated
`GroundReactionCharts.tsx` import of the missing `PlateImportForm` export. The
same TypeScript error and case collision were reproduced from untouched base
`b279eb5`. The Tauri runtime was not run, so packaged-runtime acceptance remains
unverified.
