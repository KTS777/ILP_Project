# Metagol-style Inductive Logic Programming in Python

## Overview

This project implements a simplified Metagol-style Inductive Logic Programming (ILP) system in Python.

The system learns logic programs from:
- Background knowledge (facts)
- Positive examples
- Negative examples
- A set of metarules

It constructs hypotheses by instantiating metarules and validating them using a custom SLD-resolution prover.

This implementation is designed for educational and research purposes as part of a graduation thesis.

---

## Features

- Logic term representation (Variables, Constants)
- Atoms and Clauses
- Substitution and Most General Unifier (MGU)
- Depth-limited SLD-resolution
- Standardize-apart variable renaming
- Metarule-driven hypothesis construction
- Iterative deepening over hypothesis size
- Recursive rule learning
- Negative example constraint enforcement

---

## Supported Metarules

The system currently supports:

### Identity
P(A,B) :- Q(A,B)

### Chain
P(A,B) :- Q(A,C), R(C,B)

### Fork
P(A,B) :- Q(C,A), R(C,B)

### Tail Recursion
P(A,B) :- Q(A,C), P(C,B)

---

## How It Works

1. For each positive example:
   - If it is not provable from current hypothesis H,
   - The system selects a metarule.
   - Unifies the metarule head with the example.
   - Instantiates predicate variables.
   - Generates a candidate clause.
   - Adds it to H if:
     - It helps prove the example
     - It does not entail any negative example

2. Iterative deepening controls hypothesis size.

3. Final hypothesis must:
   - Cover all positive examples
   - Reject all negative examples

---

## Example Learning Tasks

### Grandparent
Learn:
gp(X,Y) :- parent(X,Z), parent(Z,Y).


### Ancestor (Recursive)
Learn:
ancestor(X,Y) :- parent(X,Y).
ancestor(X,Y) :- parent(X,Z), ancestor(Z,Y).


### Sibling (Fork)
Learn:
sibling(X,Y) :- parent(Z,X), parent(Z,Y).


---

## Running the Program

Simply execute: python logic.py


Test functions available:
- `test_grandparent()`
- `test_ancestor()`
- `test_sibling()`
- `test_failure()`

---

## Limitations

- No predicate invention
- No type constraints
- No inequality handling (unless explicitly added)
- Search is not optimized
- Intended for clarity rather than performance

---

## Purpose

This project demonstrates:
- Unification and logical inference
- Metarule-driven hypothesis search
- Recursive rule induction
- Symbolic learning from examples

It is a simplified but faithful implementation of the Metagol pseudocode.

