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
