if [ -d "./bookcorpus_conclusion" ]
then
    rm -r ./bookcorpus_conclusion
fi
mkdir ./bookcorpus_conclusion


python conclusion_corpus_construction.py --start 0 --end 500 &
python conclusion_corpus_construction.py --start 500 --end 1000 &
python conclusion_corpus_construction.py --start 1000 --end 1500 &
python conclusion_corpus_construction.py --start 1500 --end 2000 &
python conclusion_corpus_construction.py --start 2000 --end 2500 &
python conclusion_corpus_construction.py --start 2500 --end 3000 &
python conclusion_corpus_construction.py --start 3000 --end 3500 &
python conclusion_corpus_construction.py --start 3500 --end 4000 &
python conclusion_corpus_construction.py --start 4000 --end 4500 &
python conclusion_corpus_construction.py --start 4500 --end 5000 &
python conclusion_corpus_construction.py --start 5000 --end 5500 &
python conclusion_corpus_construction.py --start 5500 --end 6000 &
python conclusion_corpus_construction.py --start 6000 --end 6500 &
python conclusion_corpus_construction.py --start 6500 --end 7000 &
python conclusion_corpus_construction.py --start 7000 --end 7500 &
python conclusion_corpus_construction.py --start 7500 --end 8000 &
python conclusion_corpus_construction.py --start 8000 --end 8500 &
python conclusion_corpus_construction.py --start 8500 --end 9000 &
python conclusion_corpus_construction.py --start 9000 --end 9500 &
python conclusion_corpus_construction.py --start 9500 --end 10000 &
python conclusion_corpus_construction.py --start 10000 --end 10500 &
python conclusion_corpus_construction.py --start 10500 --end 11000 &
python conclusion_corpus_construction.py --start 11000 --end 11500 &
python conclusion_corpus_construction.py --start 11500 --end 12000 &
python conclusion_corpus_construction.py --start 12000 --end 12500 &
python conclusion_corpus_construction.py --start 12500 --end 13000 &
python conclusion_corpus_construction.py --start 13000 --end 13500 &
python conclusion_corpus_construction.py --start 13500 --end 14000 &
python conclusion_corpus_construction.py --start 14000 --end 14500 &
python conclusion_corpus_construction.py --start 14500 --end 15000 &
python conclusion_corpus_construction.py --start 15000 --end 15500 &
python conclusion_corpus_construction.py --start 15500 --end 16000 &
python conclusion_corpus_construction.py --start 16000 --end 16500 &
python conclusion_corpus_construction.py --start 16500 --end 17000 &
python conclusion_corpus_construction.py --start 17000 --end 17500 &
python conclusion_corpus_construction.py --start 17500 --end 18000 &

wait
cat ./bookcorpus_conclusion/*.jsonl > ./bookcorpus_conclusion.jsonl
rm -r ./bookcorpus_conclusion
