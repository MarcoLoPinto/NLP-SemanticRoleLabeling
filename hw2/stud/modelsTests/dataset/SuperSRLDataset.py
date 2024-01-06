from torch.utils.data import Dataset
import numpy as np

import json
from collections import Counter

from copy import deepcopy
import torch

class SuperSRLDataset(Dataset):
    """
    This class is needed in order to read and work with data for this homework
    """
    def __init__(self, 
        lang_data_path:str = None, 
        labels = None, 
        baselines_file_path = None, 
        split_predicates = True):
        """

        Args:
            lang_data_path (str, optional): the path to a specific train or dev json dataset file. Defaults to None.
            labels (dict, optional): a dictionary containing the labels for each part of the project (if it's equal to None, then baselines_file_path is used). Defaults to None.
            baselines_file_path (str, optional): the path for the baselines file, used to generate the labels dictionary (if labels = None). Defaults to None.
            split_predicates (bool, optional): if True, each sentence is splitted in multiple sentences, one for each predicate in the sentence. Defaults to True.
        """

        self.labels = labels if labels is not None else SuperSRLDataset.get_labels_from_baselines_file(baselines_file_path)

        self.split_predicates = split_predicates
        
        if lang_data_path is not None:
            self.data_raw = SuperSRLDataset.load_data( lang_data_path )

            if split_predicates:
                self.data = SuperSRLDataset.split_data_by_predicates( self.data_raw )
            else:
                self.data = self.data_raw

    @staticmethod
    def load_data(lang_data_path):
        """load the dataset

        Args:
            lang_data_path (str): 

        Returns:
            list: list of sentences.
        """
        f = open(lang_data_path)
        res = list(json.load( f ).values())
        f.close()
        return res

    @staticmethod
    def split_data_by_predicates(data_raw):
        """splits each sentence in the dataset into multiple sentences, one for each predicate

        Args:
            data_raw (list): the dataset obtained by load_data()

        Returns:
            list: the formatted dataset
        """
        data = []
        for sentence_raw in data_raw:
            data += SuperSRLDataset.split_sentence_by_predicates(sentence_raw)
        return data

    @staticmethod
    def split_sentence_by_predicates(sentence_raw):
        """This method duplicates the sentence multiple times, one for each predicate

        Args:
            sentence_raw (dict): the sentence to be splitted

        Returns:
            list: a list of sentences, one for each predicate in the sentence
        """
        sentences = []
        # if there won't be any roles for this sentence:
        if all(p == '_' for p in sentence_raw['predicates']): 
            return sentences
        
        for i, predicate in enumerate(sentence_raw['predicates']):
            if predicate == '_':
                continue
            sentence_copy = deepcopy(sentence_raw) # raw has: words, lemmas, dependency_heads, dependency_relations, pos_tags, predicates, roles
            
            # removing every other predicate in the phrase
            preds = sentence_raw['predicates']
            sentence_copy['predicates'] = ['_']*i + [preds[i]] + ['_']*(len(preds)-i-1)
            # adding new informations of the predicate
            sentence_copy['predicate_position'] = i
            sentence_copy['predicate_label'] = sentence_raw['predicates'][i]
            sentence_copy['predicate_word'] = sentence_raw['words'][i] 
             
            # get the roles for that particular predicate as list! (if it has the "roles" attribute)
            if 'roles' in sentence_raw:
                try:
                    roles = sentence_raw['roles'][int(i)]
                except:
                    roles = sentence_raw['roles'][str(i)]
                sentence_copy['roles'] = roles # it's a list!

            sentences.append(sentence_copy)
        return sentences

    @staticmethod
    def tokenize_and_pad_sequences(batch, tokenizer):
        """returns the tokenized batch input using the tokenizer argument

        Args:
            batch (list): a batch of samples
            tokenizer: the tokenizer used

        Returns:
            dict: the tokenized batch
        """
        return tokenizer(batch, padding="max_length")

    @staticmethod
    def get_labels_from_baselines_file(baselines_file_path="data/baselines.json"):
        """generates the labels for the project

        Args:
            baselines_file_path (str, optional): The path to the baselines file. Defaults to "data/baselines.json".

        Returns:
            dict: the labels dictionary
        """
        f = open(baselines_file_path)
        baselines = json.load(f)
        f.close()

        id_to_roles = ['_'] + [ 'agent', 'asset', 'attribute', 'beneficiary', 'cause', 'co-agent', 'co-patient', 
                                'co-theme', 'destination', 'experiencer', 'extent', 'goal', 'idiom', 'instrument', 'location', 
                                'material', 'patient', 'product', 'purpose', 'recipient', 'result', 'source', 'stimulus', 
                                'theme', 'time', 'topic', 'value'
        ]
        roles_to_id = {role:i for i,role in enumerate(id_to_roles)}

        # id_to_predicates = ['_'] + sorted(list(Counter(baselines['predicate_disambiguation'].values())))

        id_to_predicates = ['_'] + ["ABSORB","ABSTAIN_AVOID_REFRAIN","ACCOMPANY","ACCUSE","ACHIEVE","ADD","ADJUST_CORRECT","AFFECT","AFFIRM","AGREE_ACCEPT","AIR","ALLY_ASSOCIATE_MARRY","ALTERNATE","AMASS","AMELIORATE","ANALYZE","ANSWER","APPEAR","APPLY","APPROVE_PRAISE","ARGUE-IN-DEFENSE","AROUSE_WAKE_ENLIVEN","ARRIVE","ASCRIBE","ASK_REQUEST","ASSIGN-SMT-TO-SMN","ATTACH","ATTACK_BOMB","ATTEND","ATTRACT_SUCK","AUTHORIZE_ADMIT","AUTOMATIZE","AUX_MOD","BE-LOCATED_BASE","BEFRIEND","BEGIN","BEHAVE","BELIEVE","BEND","BENEFIT_EXPLOIT","BETRAY","BEWITCH","BID","BLIND","BORDER","BREAK_DETERIORATE","BREATH_BLOW","BRING","BULGE-OUT","BURDEN_BEAR","BURN","BURY_PLANT","BUY","CAGE_IMPRISON","CALCULATE_ESTIMATE","CANCEL_ELIMINATE","CARRY_TRANSPORT","CARRY-OUT-ACTION","CASTRATE","CATCH","CATCH_EMBARK","CAUSE-MENTAL-STATE","CAUSE-SMT","CAVE_CARVE","CELEBRATE_PARTY","CHANGE_SWITCH","CHANGE-APPEARANCE/STATE","CHANGE-HANDS","CHANGE-TASTE","CHARGE","CHASE","CHOOSE","CIRCULATE_SPREAD_DISTRIBUTE","CITE","CLOSE","CLOUD_SHADOW_HIDE","CO-OPT","COLOR","COMBINE_MIX_UNITE","COME-AFTER_FOLLOW-IN-TIME","COME-FROM","COMMUNE","COMMUNICATE_CONTACT","COMMUNIZE","COMPARE","COMPENSATE","COMPETE","COMPLEXIFY","CONQUER","CONSIDER","CONSUME_SPEND","CONTAIN","CONTINUE","CONTRACT-AN-ILLNESS_INFECT","CONVERT","COOK","COOL","COPY","CORRELATE","CORRODE_WEAR-AWAY_SCRATCH","CORRUPT","COST","COUNT","COURT","COVER_SPREAD_SURMOUNT","CREATE_MATERIALIZE","CRITICIZE","CRY","CUT","DANCE","DEBASE_ADULTERATE","DECEIVE","DECIDE_DETERMINE","DECREE_DECLARE","DEFEAT","DELAY","DERIVE","DESTROY","DEVELOP_AGE","DIET","DIM","DIP_DIVE","DIRECT_AIM_MANEUVER","DIRTY","DISAPPEAR","DISBAND_BREAK-UP","DISCARD","DISCUSS","DISLIKE","DISMISS_FIRE-SMN","DISTINGUISH_DIFFER","DIVERSIFY","DIVIDE","DOWNPLAY_HUMILIATE","DRESS_WEAR","DRINK","DRIVE-BACK","DROP","DRY","EARN","EAT_BITE","EMBELLISH","EMCEE","EMIT","EMPHASIZE","EMPTY_UNLOAD","ENCLOSE_WRAP","ENDANGER","ENJOY","ENTER","ESTABLISH","EXCRETE","EXEMPT","EXHAUST","EXIST_LIVE","EXIST-WITH-FEATURE","EXPLAIN","EXPLODE","EXTEND","EXTRACT","FACE_CHALLENGE","FACIAL-EXPRESSION","FAIL_LOSE","FAKE","FALL_SLIDE-DOWN","FEEL","FIGHT","FILL","FIND","FINISH_CONCLUDE_END","FIT","FLATTEN_SMOOTHEN","FLATTER","FLOW","FLY","FOCUS","FOLLOW_SUPPORT_SPONSOR_FUND","FOLLOW-IN-SPACE","FORGET","FRUSTRATE_DISAPPOINT","FUEL","GENERATE","GIVE_GIFT","GIVE-BIRTH","GIVE-UP_ABOLISH_ABANDON","GO-FORWARD","GROUND_BASE_FOUND","GROUP","GROW_PLOW","GUARANTEE_ENSURE_PROMISE","GUESS","HANG","HAPPEN_OCCUR","HARMONIZE","HAVE-A-FUNCTION_SERVE","HAVE-SEX","HEAR_LISTEN","HEAT","HELP_HEAL_CARE_CURE","HIRE","HIT","HOLE_PIERCE","HOST_MEAL_INVITE","HUNT","HURT_HARM_ACHE","IMAGINE","IMPLY","INCITE_INDUCE","INCLINE","INCLUDE-AS","INCREASE_ENLARGE_MULTIPLY","INFER","INFLUENCE","INFORM","INSERT","INTERPRET","INVERT_REVERSE","ISOLATE","JOIN_CONNECT","JOKE","JUMP","JUSTIFY_EXCUSE","KILL","KNOCK-DOWN","KNOW","LAND_GET-OFF","LAUGH","LEAD_GOVERN","LEARN","LEAVE_DEPART_RUN-AWAY","LEAVE-BEHIND","LEND","LIBERATE_ALLOW_AFFORD","LIE","LIGHT_SHINE","LIGHTEN","LIKE","LOAD_PROVIDE_CHARGE_FURNISH","LOCATE-IN-TIME_DATE","LOSE","LOWER","LURE_ENTICE","MAKE-A-SOUND","MAKE-RELAX","MANAGE","MATCH","MEAN","MEASURE_EVALUATE","MEET","MESS","METEOROLOGICAL","MISS_OMIT_LACK","MISTAKE","MOUNT_ASSEMBLE_PRODUCE","MOVE-BACK","MOVE-BY-MEANS-OF","MOVE-ONESELF","MOVE-SOMETHING","NAME","NEGOTIATE","NOURISH_FEED","OBEY","OBLIGE_FORCE","OBTAIN","ODORIZE","OFFEND_DISESTEEM","OFFER","OPEN","OPERATE","OPPOSE_REBEL_DISSENT","ORDER","ORGANIZE","ORIENT","OVERCOME_SURPASS","OVERLAP","PAINT","PARDON","PARTICIPATE","PAY","PERCEIVE","PERFORM","PERMEATE","PERSUADE","PLAN_SCHEDULE","PLAY_SPORT/GAME","POPULATE","POSSESS","PRECEDE","PRECLUDE_FORBID_EXPEL","PREPARE","PRESERVE","PRESS_PUSH_FOLD","PRETEND","PRINT","PROMOTE","PRONOUNCE","PROPOSE","PROTECT","PROVE","PUBLICIZE","PUBLISH","PULL","PUNISH","PUT_APPLY_PLACE_PAVE","QUARREL_POLEMICIZE","RAISE","REACH","REACT","READ","RECALL","RECEIVE","RECOGNIZE_ADMIT_IDENTIFY","RECORD","REDUCE_DIMINISH","REFER","REFLECT","REFUSE","REGRET_SORRY","RELY","REMAIN","REMEMBER","REMOVE_TAKE-AWAY_KIDNAP","RENEW","REPAIR_REMEDY","REPEAT","REPLACE","REPRESENT","REPRIMAND","REQUIRE_NEED_WANT_HOPE","RESERVE","RESIGN_RETIRE","RESIST","REST","RESTORE-TO-PREVIOUS/INITIAL-STATE_UNDO_UNWIND","RESTRAIN","RESULT_CONSEQUENCE","RETAIN_KEEP_SAVE-MONEY","REVEAL","RISK","ROLL","RUN","SATISFY_FULFILL","SCORE","SEARCH","SECURE_FASTEN_TIE","SEE","SEEM","SELL","SEND","SEPARATE_FILTER_DETACH","SETTLE_CONCILIATE","SEW","SHAPE","SHARE","SHARPEN","SHOOT_LAUNCH_PROPEL","SHOUT","SHOW","SIGN","SIGNAL_INDICATE","SIMPLIFY","SIMULATE","SING","SLEEP","SLOW-DOWN","SMELL","SOLVE","SORT_CLASSIFY_ARRANGE","SPEAK","SPEED-UP","SPEND-TIME_PASS-TIME","SPILL_POUR","SPOIL","STABILIZE_SUPPORT-PHYSICALLY","START-FUNCTIONING","STAY_DWELL","STEAL_DEPRIVE","STOP","STRAIGHTEN","STRENGTHEN_MAKE-RESISTANT","STUDY","SUBJECTIVE-JUDGING","SUBJUGATE","SUMMARIZE","SUMMON","SUPPOSE","SWITCH-OFF_TURN-OFF_SHUT-DOWN","TAKE","TAKE-A-SERVICE_RENT","TAKE-INTO-ACCOUNT_CONSIDER","TAKE-SHELTER","TASTE","TEACH","THINK","THROW","TIGHTEN","TOLERATE","TOUCH","TRANSLATE","TRANSMIT","TRAVEL","TREAT","TREAT-WITH/BY","TRY","TURN_CHANGE-DIRECTION","TYPE","UNDERGO-EXPERIENCE","UNDERSTAND","UNFASTEN_UNFOLD","USE","VERIFY","VIOLATE","VISIT",'WORK','WASH_CLEAN','WAIT','WARN','WELCOME','WATCH_LOOK-OUT','WIN','WORSEN','WRITE']

        predicates_to_id = {pred:i for i,pred in enumerate(id_to_predicates)}

        id_to_pos = sorted( [ e for e in list(Counter(baselines['argument_classification'].keys())) if e.isupper() ] )
        pos_to_id = {pos:i for i,pos in enumerate(id_to_pos)}

        id_to_dependency_relations = sorted( [ e for e in list(Counter(baselines['argument_classification'].keys())) if e.islower() ] )
        dependency_relations_to_id = {d:i for i,d in enumerate(id_to_dependency_relations)}

        return {
            'id_to_roles':id_to_roles, 'roles_to_id':roles_to_id, 
            'roles_pad_id': -1, 'roles_pad_token': '<pad>', 

            'id_to_predicates':id_to_predicates, 'predicates_to_id':predicates_to_id,  
            'predicates_pad_id': -1, 'predicates_pad_token': '<pad>',

            'id_to_pos':id_to_pos, 'pos_to_id':pos_to_id,  
            'pos_pad_id': -1, 'pos_pad_token': '<pad>',

            'id_to_dependency_relations':id_to_dependency_relations, 'dependency_relations_to_id':dependency_relations_to_id,
            'dependency_relations_pad_id': -1, 'dependency_relations_pad_token': '<pad>',
        }

    def sentence_roles_converter(self, sentence_roles, to = 'id'):
        """converts the roles in the sentence from id to word and vice-versa

        Args:
            sentence_roles (list): the roles in the sentece
            to (str, optional): if to='id' then it converts words to ids, otherwise the opposite. Defaults to 'id'.

        Raises:
            Exception: "to" parameter must be either 'id' or 'word'

        Returns:
            list: the encoded sentence
        """
        if not (to == 'id' or to == 'word'):
            raise Exception('Sorry, the parameter "to" must be either "id" or "word"!')

        if to == 'id':
            encoded = [ self.labels['roles_to_id'][role] 
                        if (role in self.labels['roles_to_id']) and (role != self.labels['roles_pad_token'])
                        else self.labels['roles_pad_id'] 
                        for role in sentence_roles]
        else:
            encoded = [ self.labels['id_to_roles'][role] 
                        if role != self.labels['roles_pad_id']
                        else self.labels['roles_pad_token'] 
                        for role in sentence_roles]
        return encoded
    
    def sentence_pos_converter(self, sentence_pos, to = 'id'):
        """converts the part-of-speech in the sentence from id to word and vice-versa

        Args:
            sentence_roles (list): the pos in the sentece
            to (str, optional): if to='id' then it converts words to ids, otherwise the opposite. Defaults to 'id'.

        Raises:
            Exception: "to" parameter must be either 'id' or 'word'

        Returns:
            list: the encoded sentence
        """
        if not (to == 'id' or to == 'word'):
            raise Exception('Sorry, the parameter "to" must be either "id" or "word"!')

        if to == 'id':
            encoded = [ self.labels['pos_to_id'][role] 
                        if (role in self.labels['pos_to_id']) and (role != self.labels['pos_pad_token'])
                        else self.labels['pos_pad_id'] 
                        for role in sentence_pos]
        else:
            encoded = [ self.labels['id_to_pos'][role] 
                        if role != self.labels['pos_pad_id']
                        else self.labels['pos_pad_token'] 
                        for role in sentence_pos]
        return encoded

    def sentence_predicates_converter(self, sentence_preds, to = 'id'):
        """converts the predicates in the sentence from id to word and vice-versa

        Args:
            sentence_roles (list): the predicates in the sentece
            to (str, optional): if to='id' then it converts words to ids, otherwise the opposite. Defaults to 'id'.

        Raises:
            Exception: "to" parameter must be either 'id' or 'word'

        Returns:
            list: the encoded sentence
        """
        if not (to == 'id' or to == 'word'):
            raise Exception('Sorry, the parameter "to" must be either "id" or "word"!')
        vocab_type = 'predicates_to_id' if to == 'id' else 'id_to_predicates'
        pad_type = self.labels['predicates_pad_id'] if to == 'id' else self.labels['predicates_pad_token']
        pad_othertype = self.labels['predicates_pad_id'] if to != 'id' else self.labels['predicates_pad_token']

        encoded = [ self.labels[vocab_type][role] 
                    if (
                        (to == 'id' and role in self.labels[vocab_type]) or 
                        (to == 'word' and role >= 0 and role < len(self.labels[vocab_type]))
                    ) and (role != pad_othertype)
                    else pad_type 
                    for role in sentence_preds]
        return encoded

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @staticmethod
    def save_dict(file_path, dict_values):
        """saves the variable in a file

        Args:
            file_path (str): save file path
            dict_values (any): the value
        """
        np.save(file_path, dict_values)

    @staticmethod
    def load_dict(file_path):
        """returns the loaded variable

        Args:
            file_path (str): saved file path

        Returns:
            any: the variable
        """
        return np.load(file_path, allow_pickle=True).tolist()

