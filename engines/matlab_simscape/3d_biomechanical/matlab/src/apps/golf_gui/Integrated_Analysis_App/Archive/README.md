# Archive

This directory contains previous versions and experimental implementations that are no longer in active use but are kept for reference and potential recovery.

## Files

### EmbeddedSkeletonPlotter.m

- **Date Archived:** October 28, 2025
- **Reason:** Simplified version with poor graphics quality compared to original SkeletonPlotter
- **Issue:** Only drew basic line segments instead of proper 3D cylinders, spheres, and materials
- **Why Kept:** May be useful reference for future embedded implementations
- **Replaced By:** Tab 3 now launches the full original SkeletonPlotter in a separate window

## Recovery

If you need to restore any of these files:

```matlab
% Copy from archive back to parent directory
copyfile('Archive/EmbeddedSkeletonPlotter.m', 'EmbeddedSkeletonPlotter.m');
```

Or using git:

```bash
git log -- Archive/EmbeddedSkeletonPlotter.m  # See history
git show <commit>:path/to/file > recovered_file.m  # Recover specific version
```

## Notes

- Files here are not part of the active codebase
- They are not tested or maintained
- Use for reference only
- Check git history for full context
