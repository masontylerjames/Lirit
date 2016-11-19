# Lirit

11/17
  - the model can overfit a single timeslice to one prediction very well, it has difficulty overfitting to a song
  - so all the predicted probabilities go towards 0, but the values that are supposed to be 1 have an average Zscore of 1.66. Notes that are supposed to be on have an average Zscore of 1.87, and notes that are supposed to be actuated have an average zscore of 1.24. Zscores are calculated individually for each time step
  - this could be workable for composition

  - Check if the model learns the same amount if only inputting one column


  Pipeline
    - midi files
    - statematrices (getting to here is totally fine at this point)
    - inputs and ouptuts for fitting
      - any feature engineering is done at this point
    - fit neural net
    - compose!

  Composition Loop
    - generate a composition seed (looks like a statematrix)
    - generate an input seed by feature engineering the composition seed
    - loop:
      - make prediction based on seed
      - clean prediction into a state
      - append clean prediction to composition (just the seed first time through the loop)
      - generate an input seed by feature engineering the composition

  Architecture Ideas
    - could parallelize notes for a layer
    - could do shared weights for on/off and actuation
    - use 2d convolution on each time slice individually to do pseudo feature engineering
    - take beats as an input

11/18
  Things to try
  - provide beat to the model in some way
  - provide pitch class to the model in some way
  - provide pitch information to the model
  - the model could learn the idea of a key eventually, maybe through convolution, can that be bootstrapped with initial weights on a convolution?
  - get some domain knowledge
  - get the model some idea that low notes and high notes are treated differently
  - Could 3D convolution be useful?
  - There's always the brute force approach of adding more nodes, but that's not clever
  - I'm doing shared weights lstm on on/off and actuation, I could do lstm on them individually and merge the results of those in
  - Go for a much deeper network and use ReLUs because of the sparse data
  - final activation as hard sigmoid? can go to 0 and 1 so sum of the partials could go to 0 Problem: if something that is supposed to be 1 goes to 0 the error becomes infinite
  - why do I even keep the dep folder?
  - I should move the failed models to the dep folder, get a nice graveyard going
  - definitely think about ReLus
