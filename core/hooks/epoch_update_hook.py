
from mmcv.runner import HOOKS, Hook

@HOOKS.register_module()
class EpochUpdateHook(Hook):
    """
    Custom hook to update the current epoch in the backbone.
    """
    def __init__(self):
        pass

    def before_train_epoch(self, runner):
        """
        Called before each training epoch.
        """
        epoch = runner.epoch
        model_module = runner.model.module if hasattr(runner.model, 'module') else runner.model
        
        if hasattr(model_module, 'backbone') and hasattr(model_module.backbone, 'set_epoch'):
            model_module.backbone.set_epoch(epoch)
