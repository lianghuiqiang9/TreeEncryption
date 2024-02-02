use std::fs::File;
use std::io::BufWriter;
use std::time::Instant;
use std::usize;
use serde::Deserialize;
use concrete_core::backends::core::private::crypto::encoding::Plaintext;
use bitvec::prelude::*;
use serde::Serialize;
use crate::*;
use std::collections::HashMap;
use concrete_core::backends::core::private::crypto::bootstrap::FourierBuffers;
use concrete_core::backends::core::private::crypto::encoding::PlaintextList;
use num_traits::identities::One;
use num_traits::identities::Zero;
use concrete_core::backends::core::private::math::tensor::AsRefTensor;
use rand::distributions::{Distribution, Uniform};


#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Node {
    pub threshold: usize,
    pub feature: usize,
    pub leaf: usize,
    pub node_type: usize,  // node_type = 1 internal, node_type = 0 leaf
    pub child_vec:Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct CipherNode {
    pub threshold_ct: Vec<Vec<RLWECiphertext>>,
    pub leaf_ct: Vec<RGSWCiphertext>,
    pub node_type_ct: RGSWCiphertext,
    pub not_node_type_ct: RGSWCiphertext,
    pub child_vec_ct:Vec<RGSWCiphertext>,
}



pub fn encrypt_threshold_vec(sk: &RLWESecretKey, t: usize,index: usize,max_feature_index_add_one:usize, ctx: &mut Context) -> Vec<Vec<RLWECiphertext>>{
    let mut pt = PlaintextList::allocate(Scalar::zero(), ctx.plaintext_count());
    let logn = log2(ctx.poly_size.0);
    let mut out = Vec::new();
    for i in 0..max_feature_index_add_one{
        let mut tmp = Vec::new();
        for level in 1..=ctx.level_count.0 {
            assert!(t < ctx.poly_size.0);
            let mut ct = RLWECiphertext::allocate(ctx.poly_size);
            if i==index {
                let shift: usize = (Scalar::BITS as usize) - ctx.base_log.0 * level - logn;
                pt.as_mut_tensor().fill_with_element(Scalar::zero());
                *pt.as_mut_polynomial().get_mut_monomial(MonomialDegree(t)).get_mut_coefficient() = Scalar::one() << shift;
                sk.encrypt_rlwe(&mut ct, &pt, ctx.std, &mut ctx.encryption_generator);
            }else{
                sk.encrypt_rlwe(&mut ct, &pt, ctx.std, &mut ctx.encryption_generator);
            }
            tmp.push(ct);
        }
        out.push(tmp);
    }    
    out
}


pub fn decrypt_and_recompose(sk: &RLWESecretKey, cts: &Vec<RLWECiphertext>, ctx: &Context) -> Scalar {
    let mut bv: BitVec<Scalar, Lsb0> = BitVec::new();
    let mut pt = PlaintextList::allocate(Scalar::zero(), ctx.plaintext_count());
    for ct in cts {
        sk.binary_decrypt_rlwe(&mut pt, ct);
        match pt.as_tensor().first() {
            0 => bv.push(false),
            1 => bv.push(true),
            _ => panic!("expected binary plaintext"),
        }
    }
    bv.load::<Scalar>()
}



pub fn max_feature_index_add_one(network:& Vec<Vec<Box<Node>>>) -> usize{
    let mut max_feature_index:usize=0;
    for node_vec in network{
        for node in node_vec{
            if node.feature>max_feature_index {
                max_feature_index = node.feature;
            }
        }
    }
    max_feature_index = max_feature_index + 1;
    max_feature_index
}

pub fn max_leaf(network:& Vec<Vec<Box<Node>>>) -> usize{
    let mut max_leaf:usize=0;
    for node_vec in network{
        for node in node_vec{
            if node.leaf > max_leaf {
                max_leaf = node.leaf;
            }
        }
    }
    max_leaf
}

pub fn max_network_leaf_len(network:& Vec<Vec<Box<Node>>>) -> usize {
    let max_leaf_bits = ((max_leaf(network) + 1) as f64).log2().ceil() as usize;
    max_leaf_bits
}


pub fn create_the_rand_network(depth:usize,width:usize,logwidth:usize,leaf_len:usize, data_len:usize,feature_range:usize) -> Vec<Vec<Box<Node>>> {
    /*
    {
    let depth = 7;
        let width = 32;// 2^x
        let logwidth = (width as f32).log2() as usize;
        let leaf_len:usize = 1;
        let data_len: usize = 11;
        let feature_range = 10; //0..10 not include 10
        let simple = create_the_rand_network(depth, width, logwidth, leaf_len, data_len, feature_range);  
        save_network_to_json(simple.clone(), "./data/network_32width/model.json");
    }
    */
    
    let mut rng = rand::thread_rng();
    let leaf_rand_gen = Uniform::from(0..(1<<leaf_len));
    let data_rand_gen = Uniform::from(0..(1<<data_len));
    let feature_range_rand_gen = Uniform::from(0..feature_range);
    let half_width_rand_gen = Uniform::from(0..(width / 2));
    let node_type_rand_gen = Uniform::from(0..2);


    let mut root : Vec<Vec<Box<Node>>>=Vec::new();

    for i in 0..logwidth {

        let mut node_line : Vec<Box<Node>> = Vec::new();
        for j in 0..(1<<i){
            println!("i: {}, j : {}",i,j);
            let node = Box::new(Node {
                threshold:{ data_rand_gen.sample(& mut rng) },
                feature:{ feature_range_rand_gen.sample(&mut rng) },
                leaf:{ leaf_rand_gen.sample(&mut rng) },
                node_type:{ 1 },
                child_vec:{
                    let mut child_vec_temp = vec![0;width];
                    let left = half_width_rand_gen.sample(&mut rng);
                    let right = half_width_rand_gen.sample(&mut rng);
                    child_vec_temp[left] = 1;
                    child_vec_temp[width / 2 + right] = 1;
                    child_vec_temp
                },
            });
            node_line.push(node);
        }
        root.push(node_line);
    }

    for i in logwidth..depth {

        let mut node_line : Vec<Box<Node>> = Vec::new();
        for j in 0..width{
            println!("i: {}, j : {}",i,j);
            let node = Box::new(Node {
                threshold:{ data_rand_gen.sample(& mut rng) },
                feature:{ feature_range_rand_gen.sample(&mut rng) },
                leaf:{ leaf_rand_gen.sample(&mut rng) },
                node_type:{ node_type_rand_gen.sample(&mut rng) },
                child_vec:{
                    let mut child_vec_temp = vec![0;width];
                    let left = half_width_rand_gen.sample(&mut rng);
                    let right = half_width_rand_gen.sample(&mut rng);
                    child_vec_temp[left] = 1;
                    child_vec_temp[width / 2 + right] = 1;
                    child_vec_temp
                },
            });
            node_line.push(node);
        }
        root.push(node_line);
    }



    root
}

pub fn save_network_to_json(root:Vec<Vec<Box<Node>>>, path:& str){
    let file = File::create(path).unwrap();
    let writer = BufWriter::new(file);
    serde_json::to_writer(writer, &root).unwrap();
}

pub fn print_network(root:& Vec<Vec<Box<Node>>>,width:usize){
    let mut w=width;
    for node_vec in root {
        w = w/2;
        for node in node_vec{
            print!("{}","                  ".repeat(w));
            print!(" (t:{:<4}, f:{:<2}, l:{:<1}, p:{:<1}), c:{:?}  ",node.threshold, node.feature, node.leaf, node.node_type, node.child_vec);

        }
        println!();
    }
}
pub fn print_network_without_child_vec(root:& Vec<Vec<Box<Node>>>,width:usize){
    let mut w=width;
    for node_vec in root {
        w = w/2;
        for node in node_vec{
            print!("{}","                  ".repeat(w));
            print!(" (t:{:<4}, f:{:<2}, l:{:<2}, p:{:<1}) ",node.threshold, node.feature, node.leaf, node.node_type);

        }
        println!();
    }
}
pub fn print_network_node_type(root:& Vec<Vec<Box<Node>>>,width:usize){
    let mut w=width;
    for node_vec in root {
        w = w/2;
        for node in node_vec{
            print!("{}","   ".repeat(w));
            print!(" tp:{:<2}",node.node_type);
        }
        println!();
    }
}
pub fn print_network_leaf(root:& Vec<Vec<Box<Node>>>,width:usize){
    let mut w=width;
    for node_vec in root {
        w = w/2;
        for node in node_vec{
            print!("{}","   ".repeat(w));
            print!(" l:{:<2}",node.leaf);
        }
        println!();
    }
}

pub fn print_network_control_bit_matrix(network_control_bit_matrix:& Vec<Vec<usize>>,width:usize){
    let mut w=width;
    for network_control_bit_line in network_control_bit_matrix {
        w = w/2;
        for b in network_control_bit_line{
            print!("{}","   ".repeat(w));
            print!(" b:{:<1}",b);
        }
        println!();
    }
}
pub fn print_cipher_network_control_bit_matrix(network_control_bit_matrix:& Vec<Vec<RLWECiphertext>>,width:usize,ctx:& Context,sk:& RLWESecretKey){
    let mut w=width;
    for network_control_bit_line in network_control_bit_matrix {
        w = w/2;
        for b in network_control_bit_line{
            print!("{}","   ".repeat(w));

            let mut out = PlaintextList::allocate(Scalar::zero(), ctx.plaintext_count());
            sk.binary_decrypt_rlwe(&mut out, &b);
            let res = *out.as_polynomial().get_monomial(MonomialDegree(0)).get_coefficient();

            print!(" b:{:<1}",res);
        }
        println!();
    }
}

pub fn new_network_control_bit_matrix(network:& Vec<Vec<Box<Node>>>) -> Vec<Vec<usize>>{
    let mut network_control_bit_matrix:Vec<Vec<usize>>=Vec::new();
    for node_vec in network{
        let mut temp:Vec<usize> = Vec::new();
        for _node in node_vec{
            temp.push(0);
        }
        network_control_bit_matrix.push(temp);
    }
    network_control_bit_matrix[0][0] = 1;
    network_control_bit_matrix
}

pub fn new_cipher_network_control_bit_matrix(cipher_network:&Vec<Vec<Box<CipherNode>>>,ctx: &Context) -> Vec<Vec<RLWECiphertext>>{
    let mut network_control_bit_matrix:Vec<Vec<RLWECiphertext>>=Vec::new();

    let mut rlwe_one = RLWECiphertext::allocate(ctx.poly_size);
    *rlwe_one.get_mut_body().as_mut_tensor().first_mut() = Scalar::one();
    binary_encode(rlwe_one.get_mut_body().as_mut_tensor().first_mut());

    for node_vec in cipher_network{
        let mut temp:Vec<RLWECiphertext> = Vec::new();
        for _node in node_vec{
            temp.push(rlwe_one.clone());
        }
        network_control_bit_matrix.push(temp);
    }
    network_control_bit_matrix
}

pub fn enc_network_to_cipher_network(network:& Vec<Vec<Box<Node>>>,sk: &RLWESecretKey,mut ctx: &mut Context, max_feature_index_add_one:usize, leaf_len:usize) -> Vec<Vec<Box<CipherNode>>>{
    let mut cipher_network:Vec<Vec<Box<CipherNode>>>=Vec::new();
    for node_vec in network{
        let mut cipher_node_line:Vec<Box<CipherNode>> = Vec::new();
        for node in node_vec{
                        
            let cipher_node = Box::new(CipherNode {
                threshold_ct:{
                    encrypt_threshold_vec(sk, node.threshold, node.feature, max_feature_index_add_one, ctx)
                },
                leaf_ct:{
                    let mut class_leaf = vec![RGSWCiphertext::allocate(ctx.poly_size, ctx.base_log, ctx.level_count); leaf_len];
                        for (bit, mut ct) in (node.leaf).view_bits::<Lsb0>().iter().zip(class_leaf.iter_mut()) {
                            if *bit {
                                sk.encrypt_constant_rgsw(&mut ct, &Plaintext(Scalar::one()), &mut ctx);
                            }else{
                                sk.encrypt_constant_rgsw(&mut ct, &Plaintext(Scalar::zero()), &mut ctx);
                            }
                        }
                        class_leaf
                },
                node_type_ct:{
                    let mut temp_ct= RGSWCiphertext::allocate(ctx.poly_size, ctx.base_log, ctx.level_count);
                    if node.node_type == 1 {
                        sk.encrypt_constant_rgsw(&mut temp_ct, &Plaintext(Scalar::one()), &mut ctx);
                    }else{
                        sk.encrypt_constant_rgsw(&mut temp_ct, &Plaintext(Scalar::zero()), &mut ctx);
                    }
                    temp_ct
                },
                not_node_type_ct:{
                    let mut temp_ct= RGSWCiphertext::allocate(ctx.poly_size, ctx.base_log, ctx.level_count);
                    if node.node_type == 0 {
                        sk.encrypt_constant_rgsw(&mut temp_ct, &Plaintext(Scalar::one()), &mut ctx);
                    }else{
                        sk.encrypt_constant_rgsw(&mut temp_ct, &Plaintext(Scalar::zero()), &mut ctx);
                    }
                    temp_ct
                },
                child_vec_ct:{
                    let mut child_vec_ct_temp = vec![RGSWCiphertext::allocate(ctx.poly_size, ctx.base_log, ctx.level_count); node.child_vec.len()];
                        for i in 0..node.child_vec.len() {
                            if node.child_vec[i]==1 {
                                sk.encrypt_constant_rgsw(&mut child_vec_ct_temp[i], &Plaintext(Scalar::one()), &mut ctx);
                            }else{
                                sk.encrypt_constant_rgsw(&mut child_vec_ct_temp[i], &Plaintext(Scalar::zero()), &mut ctx);
                            }
                        }
                        child_vec_ct_temp
                },
            });

            cipher_node_line.push(cipher_node);
        }
        cipher_network.push(cipher_node_line);
        
    }

    cipher_network
}

pub fn constant_rlwe_one(sk: &RLWESecretKey, mut ctx: &mut Context) -> RLWECiphertext{
    let one = ctx.gen_unit_pt();
    let mut one_ct = RLWECiphertext::allocate(ctx.poly_size);
    sk.binary_encrypt_rlwe(&mut one_ct, &one, &mut ctx);
    one_ct
}

pub fn compare_attribute_and_threshold(cipher_node: & Box<CipherNode>,
                          client_cts: &Vec<usize>,
                          neg_sk_ct: & RGSWCiphertext,
                          ksk_map: & HashMap<usize, FourierRLWEKeyswitchKey>,
                          ctx: & Context,
                          buffers: & mut FourierBuffers<Scalar>) -> RGSWCiphertext {

    let tmp=RLWECiphertext::allocate(ctx.poly_size);
    let mut cts_out= vec![tmp; ctx.level_count.0];
        for i in 0..cipher_node.threshold_ct.len(){
            for j in 0..ctx.level_count.0{
                let mut ct_tmp = cipher_node.threshold_ct[i][j].clone();
                ct_tmp.less_eq_than(client_cts[i], buffers);//E(m>=t)
                cts_out[j].update_with_add(&ct_tmp);
            }
        }
    expand_fourier(&cts_out, ksk_map, neg_sk_ct, ctx, buffers)
}


pub fn eval(network:& Vec<Vec<Box<Node>>>,x_test: &Vec<usize>, network_control_bit_matrix:&mut Vec<Vec<usize>>,logwidth : usize,depth:usize,width:usize) -> usize{
    for i in 0..logwidth{
        for j in 0..(network[i]).len(){
            let node = &network[i][j];
            let b         = network_control_bit_matrix[i][j];
            let index     = node.feature;
            let v         = x_test[index];
            let t         = node.threshold;
            let cmp = (v >= t) as usize;
            network_control_bit_matrix[i+1][2*j+1] = b * cmp;                 //b*c
            network_control_bit_matrix[i+1][2*j]   = b - network_control_bit_matrix[i+1][2*j+1];  //b*(1-c)
        }
    }
    for i in logwidth..(depth-1){
        let temp= &mut vec![0;width];
        for j in 0..width{
            let node = &network[i][j];
            let b         = network_control_bit_matrix[i][j];
            let index     = node.feature;
            let v         = x_test[index];
            let t         = node.threshold;
            let node_type        = node.node_type;
            let cmp = (v >= t) as usize;
            
            let b_mul_node_type = b * node_type;
            let right    = b_mul_node_type * cmp;   //b*c
            let left     = b_mul_node_type - right; //b*(1-c)
            for k in 0..(width/2){
                temp[k] = temp[k] + node.child_vec[k] * left;
                temp[width/2 + k] = temp[width/2 + k] + node.child_vec[width/2  + k] * right;
            }
        }
        for j in 0..width{
            network_control_bit_matrix[i+1][j] = temp[j];
        }
    }

    let mut result:usize = 0;
    for i in 0..depth {
        for j in 0..(network[i].len()){
           if network[i][j].node_type == 0 && network_control_bit_matrix[i][j] == 1 {
            result = network[i][j].leaf;
           }
        }
    }
    result
}

pub fn eval_single(network:& Vec<Vec<Box<Node>>>,x_test: &Vec<usize>) -> usize{
    let mut network_control_bit_matrix = new_network_control_bit_matrix(network);
    let depth = network.len();
    let width = network[depth-1].len();
    let logwidth = (width as f32).log2() as usize;
    eval(network, &x_test, &mut network_control_bit_matrix, logwidth, depth, width)
}

pub fn cipher_network_evaluate(cipher_network: & Vec<Vec<Box<CipherNode>>>,
        cipher_network_control_bit_matrix: &mut Vec<Vec<RLWECiphertext>>,
        x_test_input: &Vec<usize>,
        neg_sk_ct: & RGSWCiphertext,
        ksk_map: & HashMap<usize, FourierRLWEKeyswitchKey>,
        ctx: & Context,
        mut buffers: & mut FourierBuffers<Scalar>,
        logwidth : usize,
        depth:usize,
        width:usize,
        leaf_len:usize) -> Vec<RLWECiphertext>{
    
    for i in 0..logwidth{
        for j in 0..(cipher_network[i]).len(){
            let cipher_node = cipher_network[i][j].clone();
            let control_bit            = cipher_network_control_bit_matrix[i][j].clone();
            let cmp = compare_attribute_and_threshold(&cipher_node, &x_test_input, &neg_sk_ct, &ksk_map, &ctx, &mut buffers);

            let mut right = RLWECiphertext::allocate(ctx.poly_size);
            cmp.external_product_with_buf(&mut right, &control_bit, &mut buffers);
            let mut left  = control_bit;
            left.update_with_sub(&right);

            cipher_network_control_bit_matrix[i+1][2*j+1] = right;
            cipher_network_control_bit_matrix[i+1][2*j]   = left;
        }
    }

    for i in logwidth..(depth-1){
        let tmp=RLWECiphertext::allocate(ctx.poly_size);
        let temp= &mut vec![tmp;width];
        for j in 0..width{
            let cipher_node = cipher_network[i][j].clone();
            let control_bit            = cipher_network_control_bit_matrix[i][j].clone();
            let cmp = compare_attribute_and_threshold(&cipher_node, &x_test_input, &neg_sk_ct, &ksk_map, &ctx, &mut buffers);
            
            let mut control_bit_mul_node_type = RLWECiphertext::allocate(ctx.poly_size);
            cipher_node.node_type_ct.external_product_with_buf(&mut control_bit_mul_node_type, &control_bit, &mut buffers);
            let mut right=RLWECiphertext::allocate(ctx.poly_size);
            cmp.external_product_with_buf(&mut right, &control_bit_mul_node_type, &mut buffers);
            let mut left = control_bit_mul_node_type;

            left.update_with_sub(&right);

            for k in 0..(width / 2){
                let mut left_temp = RLWECiphertext::allocate(ctx.poly_size);
                cipher_node.child_vec_ct[k].external_product_with_buf(&mut left_temp, &left, &mut buffers);
                temp[k].update_with_add(&left_temp);

                let mut right_temp=RLWECiphertext::allocate(ctx.poly_size);
                cipher_node.child_vec_ct[width/2 + k].external_product_with_buf(&mut right_temp, &right, &mut buffers);
                temp[width/2 + k].update_with_add(&right_temp);
            }
        }

        for j in 0..width{
            cipher_network_control_bit_matrix[i+1][j] = temp[j].clone();
        }
    }

    let mut result = vec![RLWECiphertext::allocate(ctx.poly_size); leaf_len];

    for i in 0..depth {
        for j in 0..(cipher_network[i].len()){
            let mut result_temp = RLWECiphertext::allocate(ctx.poly_size);
            cipher_network[i][j].not_node_type_ct.external_product_with_buf(&mut result_temp, &cipher_network_control_bit_matrix[i][j], &mut buffers);

            for k in 0..leaf_len{
                let mut leaf_temp = RLWECiphertext::allocate(ctx.poly_size);
                cipher_network[i][j].leaf_ct[k].external_product_with_buf(&mut leaf_temp, &result_temp, &mut buffers);
                result[k].update_with_add(&leaf_temp);
            }

        }
    }
    result
}


pub fn cipher_network(){
    let setup_instant_global = Instant::now(); 
    println!("******************************* step 1: server begin *******************************");
    let setup_instant = Instant::now();
    let dir_tree: &str = "./data/network_8width";
    let base_path_tree = Path::new(& dir_tree);
    let model_path = base_path_tree.join("model.json");
    let model_path = fs::File::open(model_path).unwrap();
    let root : Vec<Vec<Box<Node>>> = serde_json::from_reader(model_path).expect("Failed to parse JSON");
    let depth : usize = root.len();
    let width : usize = root[depth-1].len();
    let logwidth : usize = (width as f32).log2() as usize;

    let network = &root;
    let max_feature_index_add_one = max_feature_index_add_one(network);
    println!("max_feature_index_add_one = {}",max_feature_index_add_one);

    println!("server load network time : {:?}", setup_instant.elapsed());

    println!("server print network");
    print_network_without_child_vec(network, width);
    
    println!("server generater the parameters..");
    let setup_instant = Instant::now();
    let mut ctx_server = Context::default();//生成默认参数
    let sk = ctx_server.gen_rlwe_sk();
    let neg_sk_ct = sk.neg_gsw(&mut ctx_server);//RGSW(-s)
    let mut buffers_server = ctx_server.gen_fourier_buffers();
    let ksk_map = gen_all_subs_ksk_fourier(&sk, &mut ctx_server, &mut buffers_server);
    println!( "server generate sk, RGSW(-s), ksk_map time : {:?}",setup_instant.elapsed());

    println!("server encrypt the network");
    let leaf_len = max_network_leaf_len(network);
    println!("leaf_len = {}",leaf_len);

    let cipher_network = enc_network_to_cipher_network(network, &sk, &mut ctx_server, max_feature_index_add_one,leaf_len);

    println!("server encrypt the network time : {:?}", setup_instant.elapsed());
    println!("******************************* step 1: server end *******************************");
    //RGSW(-s), ksk_map, 
    //cipher_network. 

    println!("******************************* step 2: client begin *******************************");
    println!("client generate the parameters..");
   
    let setup_instant = Instant::now();
    let ctx_client = Context::default();//生成默认参数
    let mut buffers_client = ctx_client.gen_fourier_buffers();

    println!("client generate the parameters time : {:?}", setup_instant.elapsed());

    println!("client load private data");
    let setup_instant = Instant::now();
    let dir_data = dir_tree;
    let input_size = 10;
    let base_path = Path::new(& dir_data);
    let x_test_path = base_path.join("x_test.csv");
    let x_test = parse_csv(&x_test_path);
    let attr_num = x_test[0].len();

    println!("client print data");
    println!("attr_num : {}",attr_num);
    let x_test_input: Vec<Vec<usize>> = x_test.into_iter().take(input_size).collect();
    for i in 0..x_test_input.len(){println!("Print x_test_input[{}]: {:?}",i,x_test_input[i]);}

    println!("client load data time : {:?}", setup_instant.elapsed());

    println!("client start to private evaluate..");

    let setup_instant = Instant::now();
    let mut cipher_network_control_bit_matrix = new_cipher_network_control_bit_matrix(&cipher_network,&ctx_client);
    let mut output:Vec<Vec<RLWECiphertext>> = Vec::new(); 

    for line in 0..input_size{
        println!("client_f");
        let result = cipher_network_evaluate(  & cipher_network,
                                                                    & mut cipher_network_control_bit_matrix,
                                                                    & x_test_input[line],
                                                                    & neg_sk_ct,
                                                                    & ksk_map,
                                                                    & ctx_client,
                                                                    & mut buffers_client,
                                                                    logwidth,
                                                                    depth,
                                                                    width,
                                                                    leaf_len);
        output.push(result);
        //print_cipher_network_control_bit_matrix(&cipher_network_control_bit_matrix,width,&ctx_client,&sk);

    }
    println!( "client evaluate {} line data, finish step 2 time {:?} ", input_size, setup_instant.elapsed());
    println!( "client evalutae {} line data, average finish step 2 time {:?} ", input_size, setup_instant.elapsed()/input_size.try_into().unwrap());
    let setup_instant = Instant::now();
    println!( "client send output");
    println!("*******************************step 2: client end*******************************");


    println!("*******************************step 3: server begin *******************************");
    println!("server receive the client's output");

    println!("server decrypt the output");
    let mut network_control_bit_matrix = new_network_control_bit_matrix(network);
    for i in 0..input_size{
        let actual_scalar = decrypt_and_recompose(&sk, &output[i], &ctx_client);

        let expected_scalar = eval(network,&x_test_input[i], &mut network_control_bit_matrix, logwidth, depth, width);
        //print_network_control_bit_matrix(&network_control_bit_matrix, width);
        print!(  "line = {} ", i);
        print!(  " cnwe_actual_scalar: {:?} ",actual_scalar);
        println!(" expected_scalar   : {:?}",expected_scalar);
    }


    println!("server decrypt the output time {:?} \n", setup_instant.elapsed());
    println!("overall time: {:?}", setup_instant_global.elapsed());
    println!("*******************************step 3: server end *******************************");


}