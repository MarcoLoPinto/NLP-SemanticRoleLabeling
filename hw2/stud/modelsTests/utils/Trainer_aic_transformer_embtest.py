import torch
from torch.utils.data import DataLoader

try:
    from .Trainer_aic import Trainer_aic
except: # notebooks
    from stud.modelsTests.utils.Trainer_aic import Trainer_aic

class Trainer_aic_transformer_embtest(Trainer_aic):

    def __init__(self):
        super().__init__()

    def compute_forward(self, model, sample, device, optimizer = None):
        ''' must return a dictionary with "loss" key in it '''
        # possible inputs (there are also others, like pos tags):
        input_ids = sample['input_ids'].to(device)
        token_type_ids = sample['token_type_ids'].to(device) if 'token_type_ids' in sample else None # some transformer models don't have it
        attention_mask = sample['attention_mask'].to(device)
        
        # other inputs useful:
        matrix_subwords = sample['matrix_subwords'].to(device)
        formatted_word_ids = sample['formatted_word_ids'].to(device)
        output_mask = sample['output_mask']
        pos_tags = sample['pos_tags'].to(device)

        predicates = sample['predicates'].to(device)
        predicate_position = sample['predicate_position'].to(device)

        # outputs:
        labels = sample['roles']

        if optimizer is not None:
            optimizer.zero_grad()
        
        predictions = model.forward(
            input_ids = input_ids, 
            attention_mask = attention_mask, 
            matrix_subwords = matrix_subwords,
            formatted_word_ids = formatted_word_ids,
            batch_sentence_predicates = predicates,
            batch_predicate_position = predicate_position,
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
            'predicate_position': sample['predicate_position_raw'],
            'predicate_position_formatted': predicate_position.detach().cpu().tolist(),
        }

    def compute_validation(self, final_model, valid_dataloader, device):
        ''' must return a dictionary with "labels", "predictions" and "loss" keys '''
        val_labels, val_predictions, valid_loss = self.compute_arg_iden_class_validation_predictions(final_model.model, valid_dataloader, device)
        val_labels, val_predictions = self.compute_final_arg_iden_class_validation_predictions(final_model, valid_dataloader.dataset.data_raw, device)
        return {'labels': val_labels, 'predictions': val_predictions, 'loss': valid_loss}

    #################### Extras ####################

    def compute_arg_iden_class_validation_predictions(self, model, valid_dataloader, device):
        valid_loss = 0.0
        labels = {}
        predictions = {}
        step_cumulative = 0

        model.eval()
        model.to(device)
        with torch.no_grad():
            for step, sample in enumerate(valid_dataloader):
                dict_out = self.compute_forward(model, sample, device, optimizer = None)

                sample_labels = dict_out['labels'].detach().cpu().tolist()
                
                sample_predictions = model.get_indices(dict_out['predictions']).detach().cpu().tolist()
                sample_predicate_position = dict_out['predicate_position']
                sample_output_mask = dict_out['output_mask']

                valid_loss += dict_out['loss'].tolist() if dict_out['loss'] is not None else 0

                for i, (
                    phrase_labels, 
                    phrase_predictions, 
                    phrase_predicate_position, 
                    phrase_output_mask
                ) in enumerate(
                    zip(sample_labels, sample_predictions, sample_predicate_position, sample_output_mask)
                ):
                    labels[step_cumulative+i] = {}
                    predictions[step_cumulative+i] = {}

                    phrase_labels_formatted = []
                    phrase_predictions_formatted = []

                    for mask_position, mask_value in enumerate(phrase_output_mask):
                        if mask_value == 1:
                            phrase_labels_formatted.append( phrase_labels[mask_position] )
                            phrase_predictions_formatted.append( phrase_predictions[mask_position] )

                    labels[step_cumulative+i]['roles'] = {phrase_predicate_position : phrase_labels_formatted}
                    predictions[step_cumulative+i]['roles'] = {phrase_predicate_position : phrase_predictions_formatted}

                step_cumulative += len(sample_labels)

        return labels, predictions, (valid_loss / len(valid_dataloader))

    def compute_final_arg_iden_class_validation_predictions(self, final_model, valid_dataset, device):
        labels = {}
        predictions = {}

        final_model.model.eval()
        final_model.model.to(device)
        with torch.no_grad():
            for step, phrase in enumerate(valid_dataset):
                id = 'r_'+str(step)
                labels[id] = {}
                labels[id]['roles'] = {int(i):l for i,l in phrase['roles'].items()} if 'roles' in phrase else {}
                prediction = final_model.predict(phrase)
                predictions[id] = prediction

                # testing if it's all right...
                try:
                    for i in labels[id]['roles'].keys():
                        assert len(predictions[id]['roles'][i]) == len(labels[id]['roles'][i])
                except:
                    print(phrase.keys())
                    print('words:',len(phrase['words']), i)
                    print(labels[id]['roles'])
                    print(predictions[id]['roles'])
                    raise Exception('at:',step)

        return labels, predictions