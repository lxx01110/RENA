# cd MSD

python MSD/opt_str.py --dataset amazon   --ptb_type add --rate 0.3 --p 0.4 --r 3 --T 9 --seed 0 1 2 3 4 --device cuda:5 
python MSD/opt_str.py --dataset tfinance --ptb_type add --rate 0.3 --p 0.4 --r 2 --T 3 --seed 0 1 2 3 4 --device cuda:7
python MSD/opt_str.py --dataset yelp     --ptb_type add --rate 0.3 --p 0.6 --r 9 --T 5 --seed 0 1 2 3 4 --device cuda:6
