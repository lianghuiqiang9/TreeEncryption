use cipher_network::network::*;
use cipher_network::tree::*;
use std::env;


fn main(){
    
    let args: Vec<String> = env::args().collect();
    let choice :usize = args[1].parse().expect("Invalid number");
    let dir_data:String  = args[2].parse().expect("Invalid number");
    let input_size: usize = args[3].parse().expect("Invalid number");
    println!("{},{}",dir_data,input_size);
    
    if choice == 0{
        cipher_tree(dir_data, input_size);
    } else{
        cipher_network(dir_data, input_size);
    }
    
}



