if [ -d "./bookcorpus_premise" ]
then
    rm -r ./bookcorpus_premise
fi
mkdir ./bookcorpus_premise


python corpus_construction.py --start 0 --end 500 --indicator_type premise &
python corpus_construction.py --start 500 --end 1000 --indicator_type premise &
python corpus_construction.py --start 1000 --end 1500 --indicator_type premise &
python corpus_construction.py --start 1500 --end 2000 --indicator_type premise &
python corpus_construction.py --start 2000 --end 2500 --indicator_type premise &
python corpus_construction.py --start 2500 --end 3000 --indicator_type premise &
python corpus_construction.py --start 3000 --end 3500 --indicator_type premise &
python corpus_construction.py --start 3500 --end 4000 --indicator_type premise &
python corpus_construction.py --start 4000 --end 4500 --indicator_type premise &
python corpus_construction.py --start 4500 --end 5000 --indicator_type premise &
python corpus_construction.py --start 5000 --end 5500 --indicator_type premise &
python corpus_construction.py --start 5500 --end 6000 --indicator_type premise &
python corpus_construction.py --start 6000 --end 6500 --indicator_type premise &
python corpus_construction.py --start 6500 --end 7000 --indicator_type premise &
python corpus_construction.py --start 7000 --end 7500 --indicator_type premise &
python corpus_construction.py --start 7500 --end 8000 --indicator_type premise &
python corpus_construction.py --start 8000 --end 8500 --indicator_type premise &
python corpus_construction.py --start 8500 --end 9000 --indicator_type premise &
python corpus_construction.py --start 9000 --end 9500 --indicator_type premise &
python corpus_construction.py --start 9500 --end 10000 --indicator_type premise &
python corpus_construction.py --start 10000 --end 10500 --indicator_type premise &
python corpus_construction.py --start 10500 --end 11000 --indicator_type premise &
python corpus_construction.py --start 11000 --end 11500 --indicator_type premise &
python corpus_construction.py --start 11500 --end 12000 --indicator_type premise &
python corpus_construction.py --start 12000 --end 12500 --indicator_type premise &
python corpus_construction.py --start 12500 --end 13000 --indicator_type premise &
python corpus_construction.py --start 13000 --end 13500 --indicator_type premise &
python corpus_construction.py --start 13500 --end 14000 --indicator_type premise &
python corpus_construction.py --start 14000 --end 14500 --indicator_type premise &
python corpus_construction.py --start 14500 --end 15000 --indicator_type premise &
python corpus_construction.py --start 15000 --end 15500 --indicator_type premise &
python corpus_construction.py --start 15500 --end 16000 --indicator_type premise &
python corpus_construction.py --start 16000 --end 16500 --indicator_type premise &
python corpus_construction.py --start 16500 --end 17000 --indicator_type premise &
python corpus_construction.py --start 17000 --end 17500 --indicator_type premise &
python corpus_construction.py --start 17500 --end 18000 --indicator_type premise &

wait
cat ./bookcorpus_premise/*.jsonl > ./premise.jsonl
rm -r ./bookcorpus_premise
