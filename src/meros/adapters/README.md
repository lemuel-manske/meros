# Adapters

Adapters handle filesystem persistence and translate stored data into domain
objects. Processing and other application modules must use the public storage
methods rather than constructing paths to media or metadata themselves.

## Conventions

- Use the module-level `media_paths` and `metadata_paths` objects inside
  `LocalFsMediaStore` and `LocalFsMetadataStore`. Do not introduce
  `self.paths` or per-instance path holders.
- Keep storage methods direct and simple: resolve a path, read or write data,
  and construct the required domain objects.
- Do not add validation of IDs, coordinates, JSON schemas, field values, or
  other business rules to `fs_storage.py`. Let invalid inputs or malformed
  data fail naturally during normal operations rather than building validation
  layers in the adapter.
- Preserve straightforward file-handling behavior such as returning an empty
  collection or `None` for optional metadata that does not exist, and atomic
  writes where used.
- Do not update READMEs for temporary experiments or each small code change.
  Documentation should describe stable architecture, established conventions,
  and decisions intended to remain in place.

These are repository conventions for future adapter changes, not instructions
to add extra machinery or abstractions.
