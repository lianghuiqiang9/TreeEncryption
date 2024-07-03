
# Terminal commands

cargo build --release

cargo run --release 0 ./data/heart_11bits 10

cargo run --release 0 ./data/breast_11bits 10

cargo run --release 0 ./data/spam_11bits 10

cargo run --release 0 ./data/steel_11bits 10

cargo run --release 1 ./data/network_8width 10

cargo run --release 1 ./data/network_16width 10

cargo run --release 1 ./data/network_breast_11bits_7depth_16width 10

cargo run --release 1 ./data/network_heart_11bits_3depth_8width 10

# Question 1
cargo build --release 
Building [=======================>   ] 73/79: concrete-fftw-sys(build) time is very long.

# Acknowledge
# Thanks the https://github.com/KULeuven-COSIC/SortingHat.git 
# We use some function of rgsw and rlwe in their program.
