"""Fix double-filtering of vis_index (dataset already filters keyframes) and
name output files by the real scene index. Idempotent."""
P = 'visualize.py'
src = open(P).read()

OLD = """    vis_index_set = set(args.vis_index) if args.vis_index else None
    n_done = 0
    with torch.no_grad():
        for i_iter_val, data in enumerate(val_dataset_loader):
            if vis_index_set is not None:
                if i_iter_val not in vis_index_set:
                    continue
            elif n_done >= args.num_samples:
                break
            n_done += 1
"""
NEW = """    n_done = 0
    with torch.no_grad():
        for i_iter_val, data in enumerate(val_dataset_loader):
            if not args.vis_index and n_done >= args.num_samples:
                break
            n_done += 1
            tag = args.vis_index[i_iter_val] if args.vis_index else i_iter_val
"""
if OLD in src:
    src = src.replace(OLD, NEW, 1)
    print('loop fixed')
elif 'tag = args.vis_index' in src:
    print('loop already fixed')
else:
    print('WARN loop pattern not found')

# rename file tags from i_iter_val -> tag
for a, b in [("f'val_{i_iter_val}_cam'", "f'val_{tag}_cam'"),
             ("f'val_{i_iter_val}_pred'", "f'val_{tag}_pred'"),
             ("f'val_{i_iter_val}_gt'", "f'val_{tag}_gt'"),
             ("f'val_{i_iter_val}_gaussian'", "f'val_{tag}_gaussian'")]:
    if a in src:
        src = src.replace(a, b)

open(P, 'w').write(src)
print('done, uses tag:', 'val_{tag}_gaussian' in src)
