# Adapter rules

- Use module-level `media_paths` and `metadata_paths` directly. Do not use `self.paths`.
- Keep filesystem adapters limited to path resolution, reading, writing, and mapping persisted data to domain objects.
- Do not add validation for identifiers, coordinates, JSON schemas, values, or business rules in `fs_storage.py`.
- Preserve existing behavior for missing optional files and atomic writes.
- Other modules must access stored media and metadata through adapter methods, not construct filesystem paths themselves.
