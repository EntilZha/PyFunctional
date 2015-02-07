class FunctionalSequence(object):
    def __init__(self, sequence):
        if isinstance(sequence, list) or isinstance(sequence, dict):
            self.sequence = sequence
        else:
            self.sequence = list(sequence)

    def __eq__(self, other):
        return self.sequence == other

    def __ne__(self, other):
        return self.sequence != other

    def __hash__(self):
        return hash(self.sequence)

    def __repr__(self):
        return repr(self.sequence)

    def __str__(self):
        return str(self.sequence)

    def __unicode__(self):
        return unicode(self.sequence)

    def __format__(self, formatstr):
        return format(self.sequence, formatstr)

    def __nonzero__(self):
        return self.size() != 0

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return FunctionalSequence(self.sequence[key])
        return self.sequence[key]

    def __setitem__(self, key, value):
        self.sequence[key] = value

    def __delitem__(self, key):
        del self.sequence[key]

    def __iter__self(self):
        return iter(self.sequence)

    def __reversed__(self):
        return FunctionalSequence(reversed(self.sequence))

    def __contains__(self, item):
        return self.sequence.__contains__(item)

    def head(self):
        return FunctionalSequence(self.sequence[0])

    def first(self):
        return FunctionalSequence(self.sequence[0])

    def last(self):
        return FunctionalSequence(self.sequence[-1])

    def tail(self):
        return FunctionalSequence(self.sequence[-1])

    def drop(self, n):
        return FunctionalSequence(self.sequence[n:])

    def take(self, n):
        return FunctionalSequence(self.sequence[:n])

    def map(self, f):
        return FunctionalSequence(map(f, self.sequence))

    def filter(self, f):
        return FunctionalSequence(filter(f, self.sequence))

    def filter_not(self, f):
        g = lambda x: not f(x)
        return FunctionalSequence(filter(g, self.sequence))

    def reduce(self, f):
        return reduce(f, self.sequence)

    def count(self):
        return len(self.sequence)

    def len(self):
        return len(self.sequence)

    def size(self):
        return len(self.sequence)

    def reverse(self):
        return reversed(self)

    def distinct(self):
        return FunctionalSequence(list(set(self.sequence)))

    def any(self):
        return any(self.sequence)

    def all(self):
        return all(self.sequence)

    def enumerate(self, start=0):
        return FunctionalSequence(enumerate(self.sequence, start=start))

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

    def flat_map(self, f):
        l = []
        for e in self.sequence:
            l.extend(f(e))
        return FunctionalSequence(l)

    def group_by(self, f):
        result = {}
        for e in self.sequence:
            if result.get(f(e)):
                result.get(f(e)).append(e)
            else:
                result[f(e)] = [e]
        return FunctionalSequence(result)

    def empty(self):
        return len(self.sequence) == 0

    def non_empty(self):
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
        p2 = self.filter_not(f)
        return p1, p2

    def product(self):
        return self.reduce(lambda x, y: x * y)

    def slice(self, start, until):
        return FunctionalSequence(self.sequence[start:until])

    def sum(self):
        return sum(self.sequence)

    def set(self):
        return set(self.sequence)

    def zip(self, sequence):
        return FunctionalSequence(zip(self.sequence, sequence))

    def zip_with_index(self):
        return FunctionalSequence(list(enumerate(self.sequence)))

    def sorted(self, comp=None, key=None, reverse=False):
        return FunctionalSequence(sorted(self.sequence, cmp=comp, key=key, reverse=reverse))

    def list(self):
        return self.sequence

    def foreach(self, f):
        for e in self.sequence:
            f(e)


def seq(l):
    return FunctionalSequence(l)