To generate the data for part1:

- Ensure Orange workflow is running & has saved pre-processed data to -> "diamondsProcessed.csv" in the same directory.
Sometimes RFE (Rank) module can deselect all features resulting in no data being passed, if this is the case please select
all files and ensure data is saved as .csv outlined above.
- Run "par1_main.py" to generate a txt file called "part1Results.txt" -> This will contain a text table with data from
the generation and running of all the models on the test data. I intentionally left this as a txt file so that data can
easily be extracted for downstream use if desired.