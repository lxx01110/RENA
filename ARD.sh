python main.py --rate 0.3 --p 0.4 --observed 2.0 --imputed 1.0 --ptb_type add --num_heads 16 --threshold 1.4 --num_hidden 128 --r 3 --T 9 --device cuda:5 --dataset amazon --loss_fn sce 
python main.py --rate 0.3 --p 0.4 --observed 4.0 --imputed 1.0 --ptb_type add --num_heads 32 --threshold 1.3 --num_hidden 128 --r 2 --T 3 --device cuda:6 --dataset tfinance --loss_fn sce
python main.py --rate 0.3 --p 0.6 --observed 3.0 --imputed 1.0 --ptb_type add --num_heads 16 --threshold 1.0 --num_hidden 256 --r 9 --T 5 --device cuda:6 --dataset yelp --loss_fn sce 
