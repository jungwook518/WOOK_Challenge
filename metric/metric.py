import Levenshtein as Lev
from metric.wer_utils import Code, EditDistance, Token

class ErrorRate(object):
    """
    Provides inteface of error rate calcuation.
    Note:
        Do not use this class directly, use one of the sub classes.
    """

    def __init__(self, vocab) :
        self.total_dist = 0.0
        self.total_length = 0.0
        self.vocab = vocab

    def reset(self):
        self.total_dist = 0.0
        self.total_length = 0.0

    def __call__(self, targets, y_hats):
        """ Calculating character error rate """
        dist, length = self._get_distance(targets, y_hats)
        self.total_dist += dist
        self.total_length += length
        return self.total_dist / self.total_length

    def _get_distance(self, targets, y_hats):
        """
        Provides total character distance between targets & y_hats
        Args:
            targets (torch.Tensor): set of ground truth
            y_hats (torch.Tensor): predicted y values (y_hat) by the model
        Returns: total_dist, total_length
            - **total_dist**: total distance between targets & y_hats
            - **total_length**: total length of targets sequence
        """
        total_dist = 0
        total_length = 0

        for (target, y_hat) in zip(targets, y_hats):
            s1 = self.vocab.label_to_string(target)
            s2 = self.vocab.label_to_string(y_hat)
            print('======================')
            print("Tar: ", s1)
            print("Out: ",s2)
            print('======================')
            dist, length = self.metric(s1, s2)

            total_dist += dist
            total_length += length

        return total_dist, total_length

    def metric(self, *args, **kwargs):
        raise NotImplementedError


class CharacterErrorRate(ErrorRate):
    
    def __init__(self, vocab):
        super(CharacterErrorRate, self).__init__(vocab)

    def metric(self, s1: str, s2: str):
        
        # if '_' in sentence, means subword-unit, delete '_'
        if '_' in s1:
            s1 = s1.replace('_', '')

        if '_' in s2:
            s2 = s2.replace('_', '')

        dist = Lev.distance(s2, s1)
        length = len(s1.replace(' ', ''))

        return dist, length


class WordErrorRate(ErrorRate):
    """
    Computes the Word Error Rate, defined as the edit distance between the
    two provided sentences after tokenizing to words.
    """
    def __init__(self, vocab):
        super(WordErrorRate, self).__init__(vocab)

    def metric(self, s1, s2):
        """
        Computes the Word Error Rate, defined as the edit distance between the
        two provided sentences after tokenizing to words.
        Arguments:
            s1 (string): space-separated sentence
            s2 (string): space-separated sentence
        """

        # build mapping of words to integers
        b = set(s1.split() + s2.split())
        word2char = dict(zip(b, range(len(b))))

        # map the words to a char array (Levenshtein packages only accepts
        # strings)
        w1 = [chr(word2char[w]) for w in s1.split()]
        w2 = [chr(word2char[w]) for w in s2.split()]

        dist = Lev.distance(''.join(w1), ''.join(w2))
        length = len(s1.split())

        return dist, length

