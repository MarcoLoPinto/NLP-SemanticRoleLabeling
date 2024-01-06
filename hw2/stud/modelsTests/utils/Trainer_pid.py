import torch
from torch.utils.data import DataLoader

try:
    from .Trainer import Trainer
except: # notebooks
    from stud.modelsTests.utils.Trainer import Trainer

class Trainer_pid(Trainer):

    def __init__(self):
        super().__init__()

    def init_history(self, saved_history):
        history = {}
        history['train_history'] = [] if saved_history == {} else saved_history['train_history']
        history['valid_loss_history'] = [] if saved_history == {} else saved_history['valid_loss_history']
        history['f1_pred_iden_history'] = [] if saved_history == {} else saved_history['f1_pred_iden_history']
        history['f1_pred_disamb_history'] = [] if saved_history == {} else saved_history['f1_pred_disamb_history']
        return history

    def compute_forward(self, model, sample, device, optimizer = None):
        ''' must return a dictionary with "loss" key in it '''
        raise NotImplementedError

    def compute_validation(self, final_model, valid_dataloader, device):
        ''' must return a dictionary with "labels", "predictions" and "loss" keys '''
        raise NotImplementedError

    def compute_evaluations(self, labels, predictions):
        ''' must return a dictionary of results '''
        evaluations_results = {}

        null_tag = '_' # 0 is the null tag '_'
        evaluations_results['pred_iden'] = Trainer_pid.evaluate_predicate_identification(labels, predictions, null_tag)
        evaluations_results['pred_disamb'] = Trainer_pid.evaluate_predicate_disambiguation(labels, predictions, null_tag)

        return evaluations_results

    def update_history(self, history, valid_loss, evaluations_results):
        ''' must return the updated history dictionary '''
        f1_arg_iden = evaluations_results['pred_iden']['f1']
        f1_arg_class = evaluations_results['pred_disamb']['f1']

        history['valid_loss_history'].append(valid_loss)
        history['f1_pred_iden_history'].append(f1_arg_iden)
        history['f1_pred_disamb_history'].append(f1_arg_class)

        return history

    def print_evaluations_results(self, valid_loss, evaluations_results):
        f1_arg_iden = evaluations_results['pred_iden']['f1']
        f1_arg_class = evaluations_results['pred_disamb']['f1']
        print(f'# Validation loss => {valid_loss:0.6f} | f1-score: arg_iden = {f1_arg_iden:0.6f} arg_class = {f1_arg_class:0.6f} #')

    def conditions_for_saving_model(self, history, min_score):
        ''' must return True or False '''
        return (
            history['f1_pred_disamb_history'][-1] > max([0.0] + history['f1_pred_disamb_history'][:-1]) and 
            history['f1_pred_disamb_history'][-1] > min_score
        )

        