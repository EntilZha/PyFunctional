class FunctionalSequence:
    def __init__(self, sequence):
        if isinstance(sequence, list):
            self.sequence = sequence
        else:
            self.sequence = list(sequence)

    def map(self, f):
        return FunctionalSequence(map(f, self.sequence))

    def filter(self, f):
        return FunctionalSequence(filter(f, self.sequence))

    def filterNot(self, f):
        g = lambda x: not f(x)
        return FunctionalSequence(filter(g, self.sequence))

    def reduce(self, f, initial=None):
        return reduce(f, self.sequence, initial=initial)

    def count(self):
        return len(self.sequence)

    def len(self):
        return len(self.sequence)

    def size(self):
        return len(self.sequence)

    def reverse(self):
        return reversed(self.sequence)

    def distinct(self):
        return list(set(self.sequence))

    def any(self):
        return any(self.sequence)

    def all(self):
        return all(self.sequence)

    def enumerate(self, start=0):
        return enumerate(self.sequence, start=start)

    def max(self):
        return max(self.sequence)

    def min(self):
        return min(self.sequence)

    def find(self, f):
        for e in self.sequence:
            if f(e):
                return e
        else:
            return None

    def flatMap(self, f):
        l = []
        for e in self.sequence:
            l.extend(f(e))
        return l

    def groupBy(self, f):
        result = {}
        for e in self.sequence:
            if result.get(f(e)):
                result.get(f(e)).append(e)
            else:
                result[f(e)] = [e]
        return result

    def empty(self):
        return len(self.sequence) == 0

    def nonEmpty(self):
        return len(self.sequence) != 0

    def string(self, separator):
        if self.empty():
            return ""
        first = True
        s = ""
        for e in self.sequence:
            if first:
                s += str(e)
                first = False
            else:
                s += separator + str(e)
        return s

    def partition(self, f):
        p1 = self.filter(f)
        p2 = self.filternot(f)
        return p1, p2

    def product(self):
        return self.reduce(lambda x, y: x * y)

    def repr(self):
        return self.map(repr)

    def str(self):
        return self.map(str)

    def slice(self, start, until):
        return FunctionalSequence(self.sequence[start:until])

    def sum(self, start=None):
        return sum(self.sequence, start=start)

    def set(self):
        return set(self.sequence)

    def zip(self, seq):
        return FunctionalSequence(zip(self.sequence, seq))

    def zipWithIndex(self):
        return FunctionalSequence(enumerate(self.sequence))

    def sorted(self, comp=None, key=None, reverse=False):
        return FunctionalSequence(sorted(self.sequence, cmp=comp, key=key, reverse=reverse))

    def list(self):
        return self.sequence


def seq(l):
    FunctionalSequence(l)