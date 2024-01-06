try:
    from .modelsTests.model_pred_iden_dis_part.Model_pid_transformer_simple import PredIdenDisModel
    from .modelsTests.model_arg_iden_class_part.Model_aic_transformer_simple import ArgIdenClassModel
except: # notebooks
    from stud.modelsTests.model_pred_iden_dis_part.Model_pid_transformer_simple import PredIdenDisModel
    from stud.modelsTests.model_arg_iden_class_part.Model_aic_transformer_simple import ArgIdenClassModel

from transformers import AutoTokenizer

class FinalModel():

    # STUDENT: construct here your model
    # this class should be loading your weights and vocabulary
    # MANDATORY to load the weights that can handle the given language
    # possible languages: ["EN", "FR", "ES"]
    # REMINDER: EN is mandatory the others are extras

    def __init__(
        self, 
        language: str,
        device,
        model_type: int = 0,
        root_path = '../../../../',
        tokenizer = None,
    ):
        """the wrapper model for all the steps

        Args:
            language (str): the language used for the test set.
            device (str, optional): the device in which the model needs to be loaded. Defaults to None.
            model_type (int, optional): the type of model to be initialized. 
            [ 0 = arg iden + arg class | 1 = pred disamb + arg iden + arg class | 2 = pred iden + pred disamb + arg iden + arg class ].
            root_path (str, optional): root of the current environment. Defaults to '../../../../'. Defaults to 0.
            tokenizer (any, optional): the tokenizer to use for the models. Defaults to None.
        """

        tokenizer = AutoTokenizer.from_pretrained(tokenizer) if type(tokenizer) == str else tokenizer

        if model_type not in [0,1,2]:
            raise Exception('Please, pass model_type as 0 = arg iden + arg class, 1 = pred disamb + arg iden + arg class or 2 = pred iden + pred disamb + arg iden + arg class')

        self.aic = ArgIdenClassModel(
            language=language,
            device=device,
            root_path=root_path,
            tokenizer = tokenizer,
        )

        if model_type == 1:
            self.pid = PredIdenDisModel(
                language=language,
                device=device,
                root_path=root_path,
                has_predicates_positions = True,
                tokenizer = tokenizer,
            )
        elif model_type == 2:
            self.pid = PredIdenDisModel(
                language=language,
                device=device,
                root_path=root_path,
                has_predicates_positions = False,
                tokenizer = tokenizer,
            )
        else:
            self.pid = None

    def predict(self, sentence):

        result = {}
        if self.pid is not None:
            pid_out = self.pid.predict(sentence)
            result['predicates'] = pid_out['predicates']
            sentence['predicates'] = pid_out['predicates']
        aic_out = self.aic.predict(sentence)
        result['roles'] = aic_out['roles']

        return result