class PatienceSorter():
    def __init__(self):
        self.my_stacks = []
        self.num_item = 0
    def stacks(self):
        return self.my_stacks

    def stack_count(self):
        return len(self.my_stacks)

    def item_count(self):
        return self.num_item


    def add_item(self, n):
        self.num_item+=1
        for i, s in enumerate(self.my_stacks):
            if n <= s[0]:
                s.insert(0,n)
                return self

        self.my_stacks.append([n])
        return self

    def remove_item(self):
        assert self.num_item > 0, "no more items"

        self.num_item -=1
        min_n = 999999999
        min_idx =0
        for i, s in enumerate(self.my_stacks):
            if min_n > s[0]:
                min_n = s[0]
                min_idx=i
        n = self.my_stacks[min_idx].pop(0)
        if len(self.my_stacks[min_idx]) == 0:
            try:
                self.my_stacks = self.my_stacks[:min_idx] + self.my_stacks[min_idx+1:]
            except KeyError:
                self.my_stacks = self.my_stacks[:-1]
        return n

    def add_items(self,ns):
        for n in ns:
            self.add_item(n)
        return self
    def remove_items(self):
        A = []
        while(self.num_item>0):
            A.append(self.remove_item())
        return tuple(A)

sorter=PatienceSorter()
print(sorter.stacks())
print(sorter.add_items([7, 5, 2, 1, 8, 6, 3, 9, 4]).stacks())
print(sorter.stack_count())
print(sorter.item_count())
print(sorter.remove_items())
print(sorter.stacks())
print(sorter.stack_count())
print(sorter.item_count())
