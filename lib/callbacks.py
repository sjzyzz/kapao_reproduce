
class Callbacks:
    """
    Handles all registered callbacks
    """

    _callbacks = {
        'on_pretrain_routine_end': [],
        'on_train_batch_end': [],
        'on_train_epoch_end': [],
        'on_fit_epoch_end': [],
        'on_model_save': [],
        'on_train_end': [],
    }

    def __init__(self):
        return
    
    def register_action(self, hook, name='', callback=None):
        """
        Register a new action to a callback hook

        Args:
            hook:       The callback hook name to register the action to
            name:       The name of the action
            callback:   The callback to fire
        """
        assert hook in self._callbacks, f"hook '{hook}' not found in callbacks {self._callbacks}"
        assert callable(callback), f"callback '{callback}' is not callable"
        self._callbacks[hook].append({'name': name, 'callback': callback})
    
    def run_callbacks(self, hook, *args, **kwargs):
        """
        Loop through the registered actions and fire all callbacks
        """
        for logger in self._callbacks[hook]:
            logger['callback'](*args, **kwargs)
    
    def on_pretrain_routine_end(self, *args, **kwargs):
        """
        Fires all registered callbacks at the start of each pretraining routine
        """
        self.run_callbacks('on_pretrain_routine_end', *args, **kwargs)
    
    def on_train_batch_end(self, *args, **kwargs):
        """
        Fires all registered callbacks at the end of each training batch
        """
        self.run_callbacks('on_train_batch_end', *args, **kwargs)
    
    def on_train_epoch_end(self, *args, **kwargs):
        """
        Fires all register callbacks at the end of each training epoch
        """
        self.run_callbacks('on_train_epoch_end', *args, **kwargs)
    
    def on_fit_epoch_end(self, *args, **kwargs):
        """
        Fires all register callbacks at the end of each fit (train+val) epoch
        """
        self.run_callbacks('on_fit_epoch_end', *args, **kwargs)
    
    def on_model_save(self, *args, **kwargs):
        """
        Fires all register callbacks after each model save
        """
        self.run_callbacks('on_model_save', *args, **kwargs)
    
    def on_train_end(self, *args, **kwargs):
        """
        Fires all register callbacks at the end of training
        """
        self.run_callbacks('on_train_end', *args, **kwargs)