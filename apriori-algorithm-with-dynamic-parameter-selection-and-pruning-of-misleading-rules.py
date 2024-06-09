import numpy as np
import pandas as pd
import math
import apyori
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from datetime import datetime
import matplotlib.pyplot as plt


class Apriori:
    # loading dataset 
    def __init__(self, path):
        self.dataset = pd.read_csv(path, header=None).values.tolist()
        # self.dataset = pd.read_excel(path, header=None).values.tolist()
        self.minSupport = None
        self.supportData = {}
        self.contingency_list = []
        self.contingency_table = None
        self.confidence_or_lift_table = None
        self.minConfidence = None

    def createC1(self):
        C1 = []
        for transaction in self.dataset:
            for item in transaction:
                if not [item] in C1 and item is not np.nan:
                    C1.append([str(item)])  # add new candidate to the list
        # C1.sort()
        return list(map(frozenset, C1))  # use frozen set so we
        # can use it as a key in a dict

    def minimum_support(self, supportData, ssCnt):
        for key in ssCnt:
            support = ssCnt[key] / len(self.dataset)
            supportData[key] = support
        return supportData, np.median(list(supportData.values()))

    def scanDataset(self, D, Ck):
        ssCnt = {}
        for tid in D:
            for can in Ck:
                if can.issubset(tid):
                    ssCnt[can] = ssCnt.get(can, 0) + 1
        retList = []  # C1
        supportData = {}
        if self.minSupport is None:
            self.supportData, self.minSupport = self.minimum_support(supportData, ssCnt)
            for key in ssCnt:
                support = ssCnt[key] / len(self.dataset)
                if support > self.minSupport:
                    retList.insert(0, key)
        else:
            for key in ssCnt:
                support = ssCnt[key] / len(self.dataset)
                if support > self.minSupport:
                    retList.insert(0, key)
                supportData[key] = support
        return retList, supportData

    def aprioriGen(self, Lk, k):  # creates Ck
        retList = []
        lenLk = len(Lk)
        for i in range(lenLk):
            for j in range(i + 1, lenLk):
                L1 = list(Lk[i])[:k - 2]
                L2 = list(Lk[j])[:k - 2]  # first k-2 elements
                L1.sort()
                L2.sort()
                if L1 == L2:  # if first k-2 elements are equal
                    retList.append(Lk[i] | Lk[j])  # set union
        return retList

    def apriori(self):
        C1 = self.createC1()  # creating C1
        D = list(map(set, self.dataset))  # list -> set
        L1, supportData = self.scanDataset(D, C1)  # scanning dataset
        L = [L1]
        k = 2
        while len(L[k - 2]) > 0:
            Ck = self.aprioriGen(L[k - 2], k)
            Lk, supK = self.scanDataset(D, Ck)  # scan DB to get Lk
            supportData.update(supK)
            L.append(Lk)
            k += 1
        return L, supportData

    def minimum_confidence(self, L, supportData):
        confval_rules = {}
        len_rules = []
        for i in [x * 0.1 for x in range(0, 10)]:
            rules = self.generateRules(L, supportData, minConf=i)
            confval_rules[round(i, 1)] = len(rules)
            len_rules.append(len(rules))
            # print("Confidence : % 2f, No_of_rules : %2d" % (i, len_rules[int(i * 10 - 1)]))
        # print(len_rules)
        non_zero_len_rules = list(filter(lambda x: x != 0, len_rules))
        if len(non_zero_len_rules) % 2 == 0:
            self.minConfidence = len(non_zero_len_rules) / 2 / 10
        else:
            self.minConfidence = math.ceil(len(non_zero_len_rules) / 2) / 10

    def generateRules(self, L, supportData, minConf):  # supportData is a dict coming from scanD
        bigRuleList = []
        for i in range(1, len(L)):  # only get the sets with two or more items
            for freqSet in L[i]:
                H1 = [frozenset([item]) for item in freqSet]
                if i > 1:
                    self.rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
                else:
                    self.calcConf(freqSet, H1, supportData, bigRuleList, minConf)
        return bigRuleList

    def calcConf(self, freqSet, H, supportData, brl, minConf):
        prunedH = []  # create new list to return
        for conseq in H:
            conf = supportData[freqSet] / supportData[freqSet - conseq]  # calc confidence
            if conf >= minConf:
                # print(str(freqSet - conseq) + '-->' + str(conseq) + 'conf:' + str(conf))
                brl.append(list((freqSet - conseq, conseq, conf)))
                prunedH.append(conseq)
        return prunedH

    def rulesFromConseq(self, freqSet, H, supportData, brl, minConf):
        m = len(H[0])
        if len(freqSet) > (m + 1):  # try further merging
            Hmp1 = self.aprioriGen(H, m + 1)  # create Hm+1 new candidates
            Hmp1 = self.calcConf(freqSet, Hmp1, supportData, brl, minConf)
            if len(Hmp1) > 1:  # need at least two sets to merge
                self.rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)

    def contingency(self, c1, c2):
        t11 = 0
        t10 = 0
        t01 = 0
        t00 = 0
        for tid in self.dataset:
            if c1.issubset(tid) and c2.issubset(tid):
                t11 += 1
            elif c1.issubset(tid) and not c2.issubset(tid):
                t10 += 1
            elif not c1.issubset(tid) and c2.issubset(tid):
                t01 += 1
            else:
                t00 += 1
        self.contingency_list.append(
            [str(c1), str(c2), self.supportData[frozenset(c1)] * float(len(self.dataset)),
             self.supportData[frozenset(c2)] * float(len(self.dataset)), t11,
             t10, t01, t00])

    def confidence_or_lift(self, rules):
        for rule in rules:
            c11 = set(rule[0])
            c22 = set(rule[1])
            # print(c11, c22)
            self.contingency(c11, c22)
        self.contingency_table = pd.DataFrame(self.contingency_list,
                                              columns=['item1', 'item2', '1', '2', '11', '10', '01', '00', ])
        self.contingency_table['not1'] = len(self.dataset) - self.contingency_table['1']
        self.contingency_table['not2'] = len(self.dataset) - self.contingency_table['2']
        self.confidence_or_lift_table = self.contingency_table.loc[:, ['item1', 'item2']]
        self.confidence_or_lift_table['Interestingness_measure'] = np.where(
            (self.contingency_table['11'] / self.contingency_table['1'] <
             self.contingency_table['01'] / self.contingency_table['2']),
            'Lift', 'Confidence')


def apyori_imp(path):
    # min_support (defaults to 0.1)
    # min_confidence (defaults to 0.0)
    # min_lift (defaults to 0.0)
    results = list(apyori.apriori(path))
    return len(results)


def mlxtend_imp(path):
    te = TransactionEncoder()
    te_ary = te.fit(path).transform(path)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    frequent_itemsets = apriori(df, min_support=0.1)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0)
    return len(rules)


def our_imp(path):
    apriori = Apriori(path)
    apriori.createC1()
    L, supportData = apriori.apriori()
    apriori.minimum_confidence(L, supportData)
    rules = apriori.generateRules(L, supportData, apriori.minConfidence)
    # rules_df = pd.DataFrame(rules, columns=['Item1', 'Item2', 'Confidence'])
    # apriori.confidence_or_lift(rules)
    return len(rules)


def comparison(path):
    functions = [our_imp, mlxtend_imp, apyori_imp]  # list of functions
    time_list = []
    rule_list = []
    for fn in functions:
        t = datetime.now()
        no_of_rules = fn(path)
        time = datetime.now() - t
        rule_list.append(no_of_rules)
        time_list.append(time.total_seconds())
    r_imp_time = 2.044784
    r_rules = 11
    time_list.append(r_imp_time)
    rule_list.append(r_rules)
    return time_list, rule_list


def time_comparison(implementations, time_list):
    plt.plot(implementations, time_list)
    plt.title("Time Comparison")
    plt.ylabel("seconds")
    plt.xlabel("Implementation")
    plt.show()


def rule_comparison(implementations, rule_list):
    plt.bar(implementations, rule_list)
    plt.title("No of rules Comparison")
    plt.ylabel("No of rules")
    plt.xlabel("Implementation")
    plt.show()


def main():
    path = 'TWISE1.csv'
    time_list, rule_list = comparison(path)
    implementations = ["Presented", "mlxtend", "apyori", "arules"]
    time_comparison(implementations, time_list)
    rule_comparison(implementations, rule_list)
    # print(f'Minimum Support: {apriori.minSupport}')
    # print(f'Minimum Support: {apriori.minConfidence}')
    # print(f'Rules:\n{rules_df.head()}')
    # print(f'Contingency Table:\n{apriori.contingency_table.head()}')
    # print(f'Confidence or Lift Table:\n{apriori.confidence_or_lift_table.head()}')


if __name__ == "__main__":
    main()
