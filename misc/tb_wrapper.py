from torch.utils.tensorboard import SummaryWriter
from mmengine.utils import ManagerMixin


class WrappedTBWriter(SummaryWriter, ManagerMixin):

    def __init__(self, name, use_swanlab=False, swanlab_project='GaussianAD',
                 swanlab_experiment=None, swanlab_config=None,
                 swanlab_workspace=None, **kwargs):
        if use_swanlab:
            try:
                import swanlab
                init_kwargs = dict(
                    project=swanlab_project,
                    experiment_name=swanlab_experiment,
                    config=swanlab_config or {},
                    logdir=kwargs.get('log_dir'),
                )
                if swanlab_workspace:
                    init_kwargs['workspace'] = swanlab_workspace
                swanlab.init(**init_kwargs)
                swanlab.sync_tensorboard_torch()  # patch SummaryWriter after init (0.7.x API)
                print('[SwanLab] initialized successfully.')
            except ImportError:
                print('[WARNING] swanlab not installed, skipping SwanLab init.')
            except Exception as e:
                print(f'[WARNING] SwanLab init failed: {e}, skipping.')
        SummaryWriter.__init__(self, **kwargs)
        ManagerMixin.__init__(self, name)

    def finish(self):
        """Close TensorBoard writer and SwanLab run (if active)."""
        self.close()
        try:
            import swanlab
            swanlab.finish()
        except Exception:
            pass
