import torch
from torch.utils.data import DataLoader

try:
    from .Trainer_pid import Trainer_pid
except: # notebooks
    from stud.modelsTests.utils.Trainer_pid import Trainer_pid

class Trainer_pid_transformer_simple(Trainer_pid):

    def __init__(self):
        super().__init__()

    def compute_forward(self, model, sample, device, optimizer = None):
        ''' must return a dictionary with "loss" key in it '''
        # inputs (there are also others, like pos tags):
        input_ids = sample['input_ids'].to(device)
        attention_mask = sample['attention_mask'].to(device)
        predicates_binary = sample['predicates_binary'].to(device)
        token_type_ids = sample['token_type_ids'].to(device) if 'token_type_ids' in sample else None # some transformer models don't have it
        
        # other inputs useful:
        # matrix_subwords = sample['matrix_subwords'].to(device)
        pos_tags = sample['pos_tags'].to(device)

        # infos useful for output:
        output_mask = sample['output_mask']

        # outputs:
        labels = sample['predicates']

        if optimizer is not None:
            optimizer.zero_grad()
        
        predictions = model.forward(
            input_ids = input_ids, 
            attention_mask = attention_mask,
            batch_predicates_positions = predicates_binary,
            batch_pos_tags = pos_tags,
            token_type_ids = token_type_ids,
        )
            

        predictions_flattened = predictions.reshape(-1, predictions.shape[-1]) # (batch , sentence , n_labels) -> (batch*sentence , n_labels)
        labels_flattened = labels.view(-1) # (batch , sentence) -> (batch*sentence)

        predictions_flattened = predictions_flattened.to(device)
        labels_flattened = labels_flattened.to(device)

        if model.loss_fn is not None:
            sample_loss = model.compute_loss(predictions_flattened, labels_flattened)
        else:
            sample_loss = None

        if optimizer is not None:
            sample_loss.backward()
            optimizer.step()

        return {
            'labels':labels, 
            'predictions':predictions, 
            'loss':sample_loss,
            'output_mask':output_mask,
        }

    def compute_validation(self, final_model, valid_dataloader, device):
        ''' must return a dictionary with "labels", "predictions" and "loss" keys '''
        val_labels, val_predictions, valid_loss = self.compute_pred_iden_disamb_validation_predictions(final_model.model, valid_dataloader, device)
        val_labels, val_predictions = self.compute_final_pred_iden_disamb_validation_predictions(final_model, valid_dataloader.dataset.data_raw, device)
        return {'labels': val_labels, 'predictions': val_predictions, 'loss': valid_loss}

    #################### Extras ####################

    def compute_pred_iden_disamb_validation_predictions(self, model, valid_dataloader, device):
        valid_loss = 0.0
        labels = {}
        predictions = {}
        # step_cumulative = 0

        model.eval()
        model.to(device)
        with torch.no_grad():
            for step, sample in enumerate(valid_dataloader):
                dict_out = self.compute_forward(model, sample, device, optimizer = None)

                valid_loss += dict_out['loss'].tolist() if dict_out['loss'] is not None else 0

                # Not computing labels and predictions!

        return labels, predictions, (valid_loss / len(valid_dataloader))

    def compute_final_pred_iden_disamb_validation_predictions(self, final_model, valid_dataset, device):
        labels = {}
        predictions = {}

        final_model.model.eval()
        final_model.model.to(device)
        with torch.no_grad():
            for step, phrase in enumerate(valid_dataset):
                id = 'r_'+str(step)
                labels[id] = phrase

                predictions[id] = final_model.predict(phrase)

        return labels, predictions