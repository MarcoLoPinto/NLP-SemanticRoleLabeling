import torch
from torch.utils.data import DataLoader

try:
    from .Trainer_aic import Trainer_aic
except: # notebooks
    from stud.modelsTests.utils.Trainer_aic import Trainer_aic

class Trainer_aic_lstm(Trainer_aic):

    def __init__(self):
        super().__init__()

    def compute_forward(self, model, sample, device, optimizer = None):
        ''' must return a dictionary with "loss" key in it '''
        # possible inputs (there are also others, like pos tags):
        words = torch.tensor(sample['words'], dtype=torch.long).to(device)
        # lemmas = torch.tensor(sample['lemmas'], dtype=torch.LongTensor).to(device)
        predicates = torch.tensor(sample['predicates'], dtype=torch.long).to(device)
        predicate_position = torch.tensor(sample['predicate_position'], dtype=torch.long).to(device)
        # outputs:
        labels = torch.tensor(sample['roles'], dtype=torch.long) if 'roles' in sample else torch.tensor([], dtype=torch.long)

        if optimizer is not None:
            optimizer.zero_grad()
        
        predictions = model.forward(words, predicates, predicate_position)

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

        return {'labels':labels, 'predictions':predictions, 'loss':sample_loss}

    def compute_validation(self, final_model, valid_dataloader, device):
        ''' must return a dictionary with "labels", "predictions" and "loss" keys '''
        val_labels, val_predictions, valid_loss = self.compute_arg_iden_class_validation_predictions(final_model.model, valid_dataloader, device)
        val_labels, val_predictions = self.compute_final_arg_iden_class_validation_predictions(final_model, valid_dataloader.dataset.data_raw, device)
        return {'labels': val_labels, 'predictions': val_predictions, 'loss': valid_loss}

    #################### Extras ####################

    def compute_final_arg_iden_class_validation_predictions(self, final_model, valid_dataset, device):
        labels = {}
        predictions = {}

        final_model.model.eval()
        final_model.model.to(device)
        with torch.no_grad():
            for step, phrase in enumerate(valid_dataset):

                id = str(step)
                prediction = final_model.predict(phrase)
                labels[id] = {}
                labels[id]['roles'] = {int(i):l for i,l in phrase['roles'].items()} if 'roles' in phrase else {}
                predictions[id] = prediction

        return labels, predictions

    def compute_arg_iden_class_validation_predictions(self, model, valid_dataloader, device):
        valid_loss = 0.0
        labels = {}
        predictions = {}
        step_cumulative = 0

        model.eval()
        model.to(device)
        with torch.no_grad():
            for step, sample in enumerate(valid_dataloader):
                sample_dict_out = self.compute_forward(model, sample, device, optimizer = None)
                sample_labels = sample_dict_out['labels'].detach().cpu().tolist()
                sample_predictions = model.get_indices(sample_dict_out['predictions'])
                sample_predictions = sample_predictions.detach().cpu().tolist()
                sample_predicate_position = sample['predicate_position']
                # now we have both sample_labels and 
                # sample_predictions with shape (batch , roles_for_phrase).
                # Also sample_predicate_position with shape (batch)

                valid_loss += sample_dict_out['loss'].tolist() if sample_dict_out['loss'] is not None else 0

                for i, (phrase_labels, phrase_predictions, phrase_predicate_position) in enumerate(zip(sample_labels, sample_predictions, sample_predicate_position)):
                    labels[step_cumulative+i] = {}
                    predictions[step_cumulative+i] = {}
                    phrase_labels_depadded = phrase_labels[:phrase_labels.index(-1)] if -1 in phrase_labels else phrase_labels
                    phrase_predictions_depadded = phrase_predictions[:len(phrase_labels_depadded)]
                    labels[step_cumulative+i]['roles'] = {phrase_predicate_position : phrase_labels_depadded}
                    predictions[step_cumulative+i]['roles'] = {phrase_predicate_position : phrase_predictions_depadded}

                step_cumulative += len(sample_labels)

        return labels, predictions, (valid_loss / len(valid_dataloader))