# Attention in Transformers: Concepts and Code in PyTorch

## Getting Started

- Course Link: [Attention in Transformers](https://learn.deeplearning.ai/courses/attention-in-transformers-concepts-and-code-in-pytorch/lesson/han2t/introduction)

### 1. The Main Ideas Behind Transformers and Attention

- There's a lot to be said about how ChatGPT works, but fundamentally it is based on something called **Transformer.**
- **Transformer** fundamentally they require 3 main parts:
- **Encoder**: Takes the input and converts it into a hidden representation
  - _Word Embedding_: Translates input into numbers
  - _Positional Encoding_: Keeping track of word orders
  - _Attention_: Helps establish relationships between words
    - Self-Attention:
      - It works by seeing how similar each word is to all the words in sentence, including itself.
      - It calculates the similarity between the first word, and all the other words in the sentence including itself.
  > Once the similarities are calculated, they are used to determine how the Transformer encodes the input (word).
  

### 2. The Matrix Math for Calculating Self-Attention

- The equation for calculating Self-Attention can be a little intimidating...

$$
\text{Attention}(Q, K, V) = \text{softmax} \left( \frac{QK^T}{\sqrt{d_k}} \right) V
$$

- Q stands for Query
- K stands for Key
- V stands for Value

> The terms **Query**, **Key**, and **Value** come from database terminology. 

- To summarize in Database terminology, 
  - the **Query** is the thing we are using to search the database and the computer calculates similarities between the **Query** and all of...
  - the **Keys** in the database and... 
  - the **Values** are what the database returns as the results of the search.





