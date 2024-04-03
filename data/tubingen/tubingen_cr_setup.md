Here, we setup the causal relationships using the tubginen pairs for common-sensical (CS) and anti-common-sensical (ACS) (i.e., adversarial (Adv) here) for the causal graphs in the following manner:

1. A -> B:
    - CS - We sample tubingen pairs such that A -> B
    - Adv - We sample tubegien paris such that A -> B and define it as B -> A
2. B <- A -> C:
    - CS - We sample 2 tubgigen pairs such that A -> B and A -> C
    - Adv - We sample tubigen paris such that B -> C and then sample a third variable A such that neither A -> B nor A -> C. Then, we define A -> B and A -> C.
3. A -> B -> C:
    - CS - We sample 2 tubgien pairs such that A -> B and B -> C
    - Adv - We sample tubigen paris such  that A -> C and then sample a third variavble B such that neither A -> B nor C -> B. Then, we define A -> B and B -> C (Not much different from CS we we want to test chaining/transitive property of causal relationship)
    - Adv v2 - We sample tubigen paris such  that A -> C and then sample a third variavble B such that neither A -> B nor C -> B. Then, we define C -> B and B -> A.
