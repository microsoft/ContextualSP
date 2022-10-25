indicator_type=$1
tmp_dir=./filter_${indicator_type}
if [ -d ${tmp_dir} ]
then
    rm -r ${tmp_dir}
fi
mkdir ${tmp_dir}


if [ ${indicator_type} == premise ]
then
    python filter.py --start_index 0 --end_index 500000 --indicator_type premise &
    python filter.py --start_index 500000 --end_index 1000000 --indicator_type premise &
    python filter.py --start_index 1000000 --end_index 1500000 --indicator_type premise

    # python filter.py --start_index 0 --end_index 50 --indicator_type premise &
    # python filter.py --start_index 50 --end_index 100 --indicator_type premise &
    # python filter.py --start_index 150 --end_index 200 --indicator_type premise
fi


if [ ${indicator_type} == conclusion ]
then
    python filter.py --start_index 0 --end_index 500000 --indicator_type conclusion &
    python filter.py --start_index 500000 --end_index 1000000 --indicator_type conclusion &
    python filter.py --start_index 1000000 --end_index 1500000 --indicator_type conclusion

    # python filter.py --start_index 0 --end_index 50 --indicator_type conclusion &
    # python filter.py --start_index 50 --end_index 100 --indicator_type conclusion &
    # python filter.py --start_index 150 --end_index 200 --indicator_type conclusion
fi

wait
cat ${tmp_dir}/*.jsonl > ./filter_${indicator_type}.jsonl
rm -r ${tmp_dir}
