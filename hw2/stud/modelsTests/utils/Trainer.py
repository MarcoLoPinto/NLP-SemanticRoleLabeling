import torch
from torch.utils.data import DataLoader

class Trainer():

    def __init__(self):
        pass

    def train(
        self,
        final_model,
        optimizer: torch.optim.Optimizer,
        train_dataloader: DataLoader,
        valid_dataloader: DataLoader = None,
        epochs: int = 5,
        verbose: bool = True,
        save_best = True,
        save_path_name = None,
        min_score = 0.5,
        saved_history = {},
        device = 'cpu'
    ):  
        """Training and evaluation function in order to save the best model

        Args:
            final_model (any): the ML wrapped model
            optimizer (torch.optim.Optimizer): the torch.optim.Optimizer used 
            train_dataloader (DataLoader): the train data created with torch.utils.data.Dataloader
            valid_dataloader (DataLoader, optional): the dev data created with torch.utils.data.Dataloader. Defaults to None.
            epochs (int, optional): number of maximum epochs. Defaults to 5.
            verbose (bool, optional): if True, then each epoch will print the training loss, the validation loss and the f1-score. Defaults to True.
            save_best (bool, optional): if True, then the best model that surpasses min_score will be saved. Defaults to True.
            save_path_name (str, optional): path and name for the best model to be saved. Defaults to None.
            min_score (float, optional): minimum score acceptable in order to be saved. Defaults to 0.5.
            saved_history (dict, optional): saved history dictionary from another session. Defaults to {}.
            device (str, optional): if we are using cpu or gpu. Defaults to 'cpu'.

        Returns:
            a dictionary of histories
        """

        history = self.init_history(saved_history) # override

        final_model.model.to(device)

        for epoch in range(epochs):
            losses = []
            
            final_model.model.train()

            # batches of the training set
            for step, sample in enumerate(train_dataloader):
                dict_out = self.compute_forward(final_model.model, sample, device, optimizer = optimizer) # override
                losses.append(dict_out['loss'].item())

            mean_loss = sum(losses) / len(losses)
            history['train_history'].append(mean_loss)
            
            if verbose or epoch == epochs - 1:
                print(f'Epoch {epoch:3d} => avg_loss: {mean_loss:0.6f}')
            
            if valid_dataloader is not None:

                valid_out = self.compute_validation(final_model, valid_dataloader, device) # override

                evaluations_results = self.compute_evaluations(valid_out['labels'], valid_out['predictions']) # override

                history = self.update_history(history, valid_out['loss'], evaluations_results) # override

                if verbose:
                    self.print_evaluations_results(valid_out['loss'], evaluations_results) # override

                # saving...

                if save_best and save_path_name is not None:
                    if self.conditions_for_saving_model(history, min_score): # override
                        print(f'----- Best value obtained, saving model -----')
                        final_model.model.save_weights(save_path_name)
                    
        return history


    def init_history(self, saved_history):
        ''' must return the initialized history dictionary '''
        raise NotImplementedError

    def compute_forward(self, model, sample, device, optimizer = None):
        ''' must return a dictionary with "loss" key in it '''
        raise NotImplementedError

    def compute_validation(self, final_model, valid_dataloader, device):
        ''' must return a dictionary with "labels", "predictions" and "loss" keys '''
        raise NotImplementedError

    def compute_evaluations(self, labels, predictions):
        ''' must return a dictionary of results '''
        raise NotImplementedError

    def update_history(self, history, valid_loss, evaluations_results):
        ''' must return the updated history dictionary '''
        raise NotImplementedError

    def print_evaluations_results(self, valid_loss, evaluations_results):
        print('Not implemented.')
        raise NotImplementedError

    def conditions_for_saving_model(self, history, min_score):
        ''' must return True or False '''
        raise NotImplementedError

    ########### utils evaluations! ###########

    @staticmethod
    def evaluate_predicate_identification(labels, predictions, null_tag="_"):
        true_positives, false_positives, false_negatives = 0, 0, 0
        for sentence_id in labels:
            gold_predicates = labels[sentence_id]["predicates"]
            pred_predicates = predictions[sentence_id]["predicates"]
            for g, p in zip(gold_predicates, pred_predicates):
                if g != null_tag and p != null_tag:
                    true_positives += 1
                elif p != null_tag and g == null_tag:
                    false_positives += 1
                elif g != null_tag and p == null_tag:
                    false_negatives += 1
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        f1 = 2 * (precision * recall) / (precision + recall)
        return {
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    @staticmethod
    def evaluate_predicate_disambiguation(labels, predictions, null_tag="_"):
        true_positives, false_positives, false_negatives = 0, 0, 0
        for sentence_id in labels:
            gold_predicates = labels[sentence_id]["predicates"]
            pred_predicates = predictions[sentence_id]["predicates"]
            for g, p in zip(gold_predicates, pred_predicates):
                if g != null_tag and p != null_tag:
                    if p == g:
                        true_positives += 1
                    else:
                        false_positives += 1
                        false_negatives += 1
                elif p != null_tag and g == null_tag:
                    false_positives += 1
                elif g != null_tag and p == null_tag:
                    false_negatives += 1
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        f1 = 2 * (precision * recall) / (precision + recall)
        return {
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    @staticmethod
    def evaluate_argument_identification(labels, predictions, null_tag="_"):
        true_positives, false_positives, false_negatives = 0, 0, 0
        for sentence_id in labels:
            gold = labels[sentence_id]["roles"]
            pred = predictions[sentence_id]["roles"]
            predicate_indices = set(gold.keys()).union(pred.keys())
            for idx in predicate_indices:
                if idx in gold and idx not in pred:
                    false_negatives += sum(1 for role in gold[idx] if role != null_tag)
                elif idx in pred and idx not in gold:
                    false_positives += sum(1 for role in pred[idx] if role != null_tag)
                else:  # idx in both gold and pred
                    for r_g, r_p in zip(gold[idx], pred[idx]):
                        if r_g != null_tag and r_p != null_tag:
                            true_positives += 1
                        elif r_g != null_tag and r_p == null_tag:
                            false_negatives += 1
                        elif r_g == null_tag and r_p != null_tag:
                            false_positives += 1

        # precision = true_positives / (true_positives + false_positives)
        # recall = true_positives / (true_positives + false_negatives)
        # f1 = 2 * (precision * recall) / (precision + recall)
        if (true_positives + false_positives) != 0:
            precision = true_positives / (true_positives + false_positives)
        else:
            precision = 0.0
        if (true_positives + false_negatives) != 0:
            recall = true_positives / (true_positives + false_negatives)
        else:
            recall = 0.0
        if (precision + recall) != 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
        return {
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    @staticmethod
    def evaluate_argument_classification(labels, predictions, null_tag="_"):
        true_positives, false_positives, false_negatives = 0, 0, 0
        for sentence_id in labels:
            gold = labels[sentence_id]["roles"]
            pred = predictions[sentence_id]["roles"]
            predicate_indices = set(gold.keys()).union(pred.keys())

            for idx in predicate_indices:
                if idx in gold and idx not in pred:
                    false_negatives += sum(1 for role in gold[idx] if role != null_tag)
                elif idx in pred and idx not in gold:
                    false_positives += sum(1 for role in pred[idx] if role != null_tag)
                else:  # idx in both gold and pred
                    for r_g, r_p in zip(gold[idx], pred[idx]):
                        if r_g != null_tag and r_p != null_tag:
                            if r_g == r_p:
                                true_positives += 1
                            else:
                                false_positives += 1
                                false_negatives += 1
                        elif r_g != null_tag and r_p == null_tag:
                            false_negatives += 1
                        elif r_g == null_tag and r_p != null_tag:
                            false_positives += 1

        # precision = true_positives / (true_positives + false_positives)
        # recall = true_positives / (true_positives + false_negatives)
        # f1 = 2 * (precision * recall) / (precision + recall)
        if (true_positives + false_positives) != 0:
            precision = true_positives / (true_positives + false_positives)
        else:
            precision = 0.0
        if (true_positives + false_negatives) != 0:
            recall = true_positives / (true_positives + false_negatives)
        else:
            recall = 0.0
        if (precision + recall) != 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
        return {
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }