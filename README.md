# Lirit

11/17
  - the model can overfit a single timeslice to one prediction very well, it has difficulty overfitting to a song
  - so all the predicted probabilities go towards 0, but the values that are supposed to be 1 have an average Zscore of 1.66. Notes that are supposed to be on have an average Zscore of 1.87, and notes that are supposed to be actuated have an average zscore of 1.24. Zscores are calculated individually for each time step
  - this could be workable for composition

  - Check if the model learns the same amount if only inputting one column
