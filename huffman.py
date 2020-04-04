from queue import PriorityQueue, Empty
from collections import defaultdict


class BitList(object):
    def __init__(self):
        self.acc = bytearray()
        self.tmp = ""
        self.sealed = False

    def extend(self, bits):
        assert not self.sealed
        self.tmp += bits
        while len(self.tmp) >= 8:
            self.acc.append(int(self.tmp[:8], base=2))
            self.tmp = self.tmp[8:]

    def seal(self):
        assert not self.sealed
        if self.tmp:
            end_byte_len = len(self.tmp)
            self.tmp += "0000000"
            self.acc.append(int(self.tmp[:8], base=2))
        else:
            end_byte_len = 8
        self.acc.append(end_byte_len)
        self.sealed = True

    def generator(self):
        assert self.sealed
        endbyte_len = self.acc[-1]
        assert 0 < endbyte_len <= 8
        for c in self.acc[:-2]:
            for b in "{:>08s}".format(bin(c)[2:]):
                yield b
        c = self.acc[-2]
        for b in "{:>08s}".format(bin(c)[2:])[:endbyte_len]:
            yield b

    @staticmethod
    def frombytes(b):
        res = BitList()
        res.acc = b
        res.sealed = True
        return res


class Node(object):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __lt__(self, other):
        if isinstance(self, Leaf) and isinstance(other, Leaf):
            return self.value < other.value
        elif isinstance(self, Leaf):
            return True
        elif isinstance(other, Leaf):
            return False
        else:
            if self.left < other.left:
                return True
            elif self.left == other.left:
                return self.right < other.right
            else:
                return False

    def __le__(self, other):
        if isinstance(self, Leaf) and isinstance(other, Leaf):
            return self.value <= other.value
        elif isinstance(self, Leaf):
            return True
        elif isinstance(other, Leaf):
            return False
        else:
            if self.left < other.left:
                return True
            elif self.left == other.left:
                return self.right <= other.right
            else:
                return False

    def __eq__(self, other):
        if isinstance(self, Leaf) and isinstance(other, Leaf):
            return self.value == other.value
        elif isinstance(self, Leaf):
            return False
        elif isinstance(other, Leaf):
            return False
        else:
            return self.left == other.left and self.right == other.right

    def __hash__(self):
        return hash(self.left) ^ hash(self.right)

    def bin(self):
        data, bit_list = self._bin(bytearray(), BitList())
        bit_list.seal()
        return data, bit_list.acc

    def _bin(self, data, bit_list):
        bit_list.extend("0")
        data, bit_list = self.left._bin(data, bit_list)
        data, bit_list = self.right._bin(data, bit_list)
        return data, bit_list

    @staticmethod
    def reconstruct(step, data, structure):
        root = None
        stack = []
        j = 0
        for i in BitList.frombytes(structure).generator():
            if i == "0":
                newnode = Node(None, None)
                if stack:
                    prev = stack[-1]
                    if prev.left is None:
                        prev.left = newnode
                    else:
                        assert prev.right is None
                        prev.right = newnode
                else:
                    root = newnode
                stack.append(newnode)
            else:
                assert i == "1"
                newnode = Leaf(bytes(data[j: j+step]))
                j += step
                if stack:
                    prev = stack[-1]
                    if prev.left is None:
                        prev.left = newnode
                    else:
                        assert prev.right is None
                        prev.right = newnode
                        while stack and stack[-1].right is not None:
                            stack.pop()
                else:
                    root = newnode
        assert not stack
        assert j == len(data)
        return root


class Leaf(Node):
    def __init__(self, i):
        self.value = i

    def __hash__(self):
        return hash(self.value)

    def _bin(self, data, bit_list):
        data.extend(self.value)
        bit_list.extend("1")
        return data, bit_list


def pad(s, n):
    if len(s) < n:
        return s + b"\x00" * (n - len(s))
    return s


def tokenize(s, n):
    for i in range(0, len(s)//n):
        yield s[n*i:n*(i+1)]
    if len(s) % n != 0:
        yield pad(s[n*(len(s)//n):], n)


def encode(tokenized_text, key):
    cipher_text = BitList()
    for c in tokenized_text:
        cipher_text.extend(key[c])
    cipher_text.seal()
    return cipher_text.acc


def decode(cipher_text, node):
    text = bytearray()
    curr = node
    for b in BitList.frombytes(cipher_text).generator():
        if b == "0":
            curr = curr.left
        else:
            assert b == "1"
            curr = curr.right
        if isinstance(curr, Leaf):
            text.extend(curr.value)
            curr = node
    return bytes(text)


def huffman(tokens):
    dt = defaultdict(int)
    for k in tokens:
        dt[k] += 1
    q = PriorityQueue()
    for k, v in dt.items():
        q.put((v, Leaf(k)))
    try:
        while True:
            ai, a = q.get()
            bi, b = q.get_nowait()
            c = Node(a, b)
            q.put((ai+bi, c))
    except Empty:
        pass
    return a


def get_key(root_node):
    key = {}

    def add_to_key(key, node, stack=[]):
        if isinstance(node, Leaf):
            key[node.value] = "".join(stack)
        else:
            stack.append("0")
            add_to_key(key, node.left, stack)
            stack.pop()
            stack.append("1")
            add_to_key(key, node.right, stack)
            stack.pop()
    add_to_key(key, root_node)
    return key


if __name__ == "__main__":
    for i in range(1, 4):
        ipath = "data/input{}.txt".format(i)
        opath = "output/output{}.txt".format(i)
        dpath = "output/decode{}.txt".format(i)
        with open(ipath, "rb") as f:
            src = f.read()
        root_node = huffman(tokenize(src, 2))
        key = get_key(root_node)
        with open(opath, "wb") as f:
            f.write(encode(tokenize(src, 2), key))
        with open(opath, "rb") as f:
            cipher_text = f.read()
        with open(dpath, "wb") as f:
            f.write(decode(cipher_text, root_node))
