Abstract. 
In the field of Knowledge Discovery in Databases (KDD) the effec-tiveness of association rules is important. Association Rules is a technique of data mining, wherein we identify the relationship between one item to another. For mining, the association rules Apriori algorithm is widely used. The idea of the Apriori algorithm is to find the frequent sets from a transactional database. Through the frequent sets, association rules are obtained, these rules must satisfy the minimum confidence threshold. This paper presents an improved method for deciding an optimum minimum support threshold and minimum confidence threshold, pruning of rules based on a contingency table, and finally the decision about whether to go for lift or confidence to get rid of uninteresting, misleading, and confusing association rules.
Keywords: Minimum support , Minimum confidence , Contingency table, Lift, Conviction, Leverage.

1 Introduction
In data mining, the Mining of association rule is important. How to create frequent itemsets is important. The prime aspect is to better the mining algorithm that how to deduce item sets candidate to produce frequent item sets effectively.
Market Basket Analysis contains two factors: Frequent itemsets and Frequent se-quential patterns. A frequent itemset is a collection of items that are often purchased together. In technical terms, frequent itemsets are the item sets that satisfy the minimum support threshold defined by the practitioner. For instance, a grocery store has stored a transactional database, which tells us what commodities customers buy. If a few items, for example, say, Bread and Milk, are repeated in most of the transactions, then the two commodities Bread and Milk form a frequent itemset.
In Frequent sequential patterns, to understand this pattern, assume that we visit a computer store to study the transactional patterns of customers that buy goods. It be-comes evident from multiple sales analysis that many customers follow a certain pattern while purchasing computers. For example, A consumer buys a laptop and in addition to it, will pair it with a purchase of an antivirus software system. The database suggests that this pairing of commodities occurs many times. This frequent occurrence of pat-terns in buying is termed a sequential pattern.

Support
It is the count of items or itemset occurred in transactions. It measures the frequency of items or itemset.
Support(X) = count of transactions having X item / Total number of transactions

Confidence
Confidence of a rule is calculated by dividing the probability of the items that occur together by the probability of the occurrence of the antecedent.
If X and Y are two items then
Confidence (X => Y) = support of (X, Y) / support of (X)

Lift
It gives information about the degree to which the occurrence of one item increases the probability of the occurrence of another item. Lift is used to find misleading rules.
Suppose a rule R1, which has Bread and Butter, and which derives Diapers
R1= {Bread, Butter} => {Diapers}
Lift ({Bread, Butter} => {Diapers}) = (confidence of R1) / (Support of Diapers
