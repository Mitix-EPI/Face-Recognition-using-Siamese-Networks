first_tests.csv = Check the best parameters and combinations & No data augmentation (24h)
Conclusion:
-> Eliminate little low neurons dense
-> Eliminate high dropout rate
-> Keep testing lr & CNN filters & high dense neurons


second_tests.csv = Less pixels but remove _slice + Data augmentation (24h)
-> Very low lr useless
-> Good results for tested all tested dense neurons & tested cnn filters
-> Maybe Dropout rate = 0.25 is the best param ?

third_tests.csv = Try fine-tunning (not with a lot of parameters) (15h)
-> Very slow
-> Not extraordinary better results compared to previous tests (but a bit higher)
-> I forgot to remove CNN filters tests (useless in fine-tunning test)