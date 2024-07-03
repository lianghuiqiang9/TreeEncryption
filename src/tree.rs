use serde::{Deserialize, Serialize};
use concrete_core::backends::core::private::crypto::encoding::Plaintext;
use bitvec::prelude::*;
use crate::*;
use std::time::Instant;
use std::collections::HashMap;
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
/// Comparison operation in the decision node.
pub enum Op {
    LEQ,
    GT,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
/// A node in the tree.
pub enum Node {
    Internal(Box<Internal>),
    Leaf(usize),
}

impl Node {
    /// Turn the node to an Internal, panic if it's a leaf.
    pub fn unwrap(self) -> Internal {
        match self {
            Node::Internal(x) => *x,
            Node::Leaf(_) => panic!("this is a leaf"),
        }
    }

    /// Create a tree with depth 1 (one internal node)
    pub fn new() -> Node {
        let mut processed_one_leaf = false;
        gen_full_tree(1, &mut processed_one_leaf)
    }

    /// Create a tree with depth d
    pub fn new_with_depth(d: usize) -> Node {
        let mut processed_one_leaf = false;
        gen_full_tree(d, &mut processed_one_leaf)
    }

    /// Assign a unique index to every node in DFS order.
    pub fn fix_index(&mut self) -> usize {
        match self {
            Node::Internal(internal) => fix_index(internal, 0),
            Node::Leaf(_) => panic!("this is a leaf")
        }
    }

    /// Return the flattened version of the tree.
    /// If fix_index is called prior, then the index should be ordered.
    pub fn flatten(&self) -> Vec<Internal> {
        match self {
            Node::Internal(internal) => {
                let mut out = Vec::new();
                // TODO reserve memory
                flatten_tree(&mut out, internal);
                out
            }
            Node::Leaf(_) => vec![],
        }
    }

    /// Evaluate the decision tree with a feature vector and output the final class.
    pub fn eval(&self, features: &Vec<usize>) -> usize {
        let mut out = 0;
        eval_node(&mut out, self, features, 1);
        out
    }

    /// Count the number of leaves.
    pub fn count_leaf(&self) -> usize {
        match self {
            Node::Internal(internal) => {
                internal.left.count_leaf() + internal.right.count_leaf()
            }
            Node::Leaf(_) => 1
        }
    }

    /// Count the number of internal nodes.
    pub fn count_internal(&self) -> usize {
        match self {
            Node::Internal(internal) => {
                1 + internal.left.count_internal() + internal.right.count_internal()
            }
            Node::Leaf(_) => 0
        }
    }

    /// Count the maximum depth.
    pub fn count_depth(&self) -> usize {
        match self {
            Node::Internal(internal) => {
                let l = internal.left.count_depth();
                let r = internal.right.count_depth();
                if l > r {
                    l + 1
                } else {
                    r + 1
                }
            }
            Node::Leaf(_) => 0,
        }
    }

    /// Find the maximum feature index in the tree.
    pub fn max_feature_index(&self) -> usize {
        match self {
            Node::Internal(internal) => {
                let i = internal.feature;
                i.max(internal.left.max_feature_index()).max(internal.right.max_feature_index())
            }
            Node::Leaf(_) => 0
        }
    }
    pub fn max_feature_index_add_one(&self) -> usize {
        self.max_feature_index()+1
    }

    
    pub fn print_tree(&self, level: usize) {
        match self {
            Node::Internal(internal) => {
                print!("Internal:");                
                print!(" feature: {}",internal.feature);
                print!(" threshold: {}",internal.threshold);
                print!(" index: {}", internal.index);
                println!(" op: {:?}", internal.op);
                print!("{}        left:", "  ".repeat(level));
                Node::print_tree(&internal.left, level + 4);
                print!("{}        right:", "  ".repeat(level));
                Node::print_tree(&internal.right, level + 4);
            }
            Node::Leaf(leaf) => {
                println!("  Leaf: {}", leaf);
            }
        }
    }

    pub fn max_leaf(&self) -> usize {
        match self {
            Node::Internal(internal) => {
                let l = internal.left.max_leaf();
                let r = internal.right.max_leaf();
                if l > r {
                    l
                } else {
                    r
                }
            }
            Node::Leaf(x) => *x
        }
    }
    pub fn max_leaf_len(&self) ->usize{
        let max_leaf_bits = ((self.max_leaf() + 1) as f64).log2().ceil() as usize;
        max_leaf_bits
    }
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
/// An internal node in a decision tree.
pub struct Internal {
    pub threshold: usize,
    pub feature: usize,
    pub index: usize,
    pub op: Op,
    pub left: Node,
    pub right: Node,
}

fn fix_index(node: &mut Internal, i: usize) -> usize {
    node.index = i;
    let j = match &mut node.left {
        Node::Leaf(_) => i,
        Node::Internal(left) => fix_index(left, i + 1)
    };
    match &mut node.right {
        Node::Leaf(_) => j,
        Node::Internal(right) => fix_index(right, j + 1)
    }
}

fn flatten_tree(out: &mut Vec<Internal>, node: &Internal) {
    out.push(Internal {
        threshold: node.threshold,
        feature: node.feature,
        index: node.index,
        op: node.op,
        left: Node::Leaf(0),
        right: Node::Leaf(0),
    });
    match &node.left {
        Node::Leaf(_) => (),
        Node::Internal(left) => flatten_tree(out, left),
    }
    match &node.right {
        Node::Leaf(_) => (),
        Node::Internal(right) => flatten_tree(out, right),
    }
}

fn eval_node(out: &mut usize, node: &Node, features: &Vec<usize>, b: usize) {
    match node {
        Node::Leaf(x) => {
            *out += *x * b;
        }
        Node::Internal(node) => {
            match node.op {
                Op::LEQ => {
                    if features[node.feature] >= node.threshold {
                        eval_node(out, &node.right, features, b);
                        eval_node(out, &node.left, features, b * (1 - b));
                    } else {
                        eval_node(out, &node.right, features, b * (1 - b));
                        eval_node(out, &node.left, features, b);
                    }
                }
                Op::GT => todo!(),
            }
        }
    }
}


fn gen_full_tree(d: usize, processed_one_leaf: &mut bool) -> Node {
    if d == 0 {
        if *processed_one_leaf {
            Node::Leaf(0)
        } else {
            *processed_one_leaf = true;
            Node::Leaf(1)
        }
    } else {
        Node::Internal(Box::new(Internal {
            threshold: 0,
            feature: 0,
            index: 0,
            op: Op::LEQ,
            left: gen_full_tree(d - 1, processed_one_leaf),
            right: gen_full_tree(d - 1, processed_one_leaf),
        }))
    }
}



pub fn compare_expand<'a>(flat_cipher_nodes: &'a Vec<CipherInternal>,
                          client_cts: &'a Vec<usize>,
                          neg_sk_ct: &'a RGSWCiphertext,
                          ksk_map: &'a HashMap<usize, FourierRLWEKeyswitchKey>,
                          ctx: &'a Context,
                          buffers: &'a mut FourierBuffers<Scalar>) -> impl Iterator<Item=RGSWCiphertext> + 'a {
    flat_cipher_nodes.iter().map(|cipher_node| {
        let tmp=RLWECiphertext::allocate(ctx.poly_size);
        let mut cts_out= vec![tmp; ctx.level_count.0];
            for i in 0..cipher_node.threshold_ct.len(){
                //i in 0 ..13
                for j in 0..ctx.level_count.0{
                    //j in 0..7
                    let mut ct_tmp=cipher_node.threshold_ct[i][j].clone();
                    match cipher_node.op {
                        Op::LEQ => ct_tmp.less_eq_than(client_cts[i], buffers),//E(m>=t)
                        Op::GT => todo!(),
                    }
                    cts_out[j].update_with_add(&ct_tmp);
                }
            }
        expand_fourier(&cts_out, ksk_map, neg_sk_ct, ctx, buffers)
    }) 
}

#[derive(Debug,Clone)] 
pub enum CipherNode {
    Internal(Box<CipherInternal>),
    Leaf(Box<CipherLeaf>),
}

impl CipherNode {
    pub fn enc_tree(original_node: &Node, sk: &RLWESecretKey, mut ctx: &mut Context, max_feature_index_add_one:usize, leaf_len:usize) -> CipherNode {
        match original_node {
            Node::Internal(internal) => {
                let left_copy = CipherNode::enc_tree(&internal.left, sk, ctx, max_feature_index_add_one, leaf_len);
                let right_copy =  CipherNode::enc_tree(&internal.right, sk, ctx, max_feature_index_add_one,leaf_len);
                let threshold_ct= encrypt_threshold_vec(sk, internal.threshold,internal.feature, max_feature_index_add_one, ctx);
                
                CipherNode::Internal(Box::new(CipherInternal {
                    threshold_ct,
                    index: internal.index,
                    op: internal.op,
                    left: left_copy,
                    right: right_copy,
                }))
            }
            Node::Leaf(value) => {
                CipherNode::Leaf(Box::new(CipherLeaf { 
                    class_leaf:{
                        let mut class_leaf = vec![RGSWCiphertext::allocate(ctx.poly_size, ctx.base_log, ctx.level_count); leaf_len];
                        for (bit, mut ct) in (*value).view_bits::<Lsb0>().iter().zip(class_leaf.iter_mut()) {
                            if *bit {
                                sk.encrypt_constant_rgsw(&mut ct, &Plaintext(Scalar::one()), &mut ctx);
                            }else{
                                sk.encrypt_constant_rgsw(&mut ct, &Plaintext(Scalar::zero()), &mut ctx);
                            }
                        }
                        class_leaf
                    }
                })) 
            }
        }
    }

    pub fn print_cipher_tree(&self, level: usize) {
        match self {
            CipherNode::Internal(internal) => {
                print!("Internal:");
                print!(" index: {}", internal.index);
                println!(" op: {:?}", internal.op);
                print!("{}        left:", "  ".repeat(level));
                CipherNode::print_cipher_tree(&internal.left, level + 4);
                print!("{}        right:", "  ".repeat(level));
                CipherNode::print_cipher_tree(&internal.right, level + 4);
                
            }
            CipherNode::Leaf(_) => {
                println!("  Leaf: {}", 0);//leaf
            }
        }
    }


    /// Return the flattened version of the tree.
    /// If fix_index is called prior, then the index should be ordered.
    pub fn flatten(&self) -> Vec<CipherInternal> {
        match self {
            CipherNode::Internal(internal) => {
                let mut out = Vec::new();
                // TODO reserve memory
                flatten_cipher_tree(&mut out, internal);
                out
            }
            CipherNode::Leaf(_) => vec![],
        }
    }
   
}

fn flatten_cipher_tree(out: &mut Vec<CipherInternal>, node: &CipherInternal) {
    out.push(CipherInternal {
        threshold_ct: node.threshold_ct.clone(),
        index: node.index,
        op: node.op,
        left: CipherNode::Leaf(Box::new(CipherLeaf {class_leaf: Vec::new()})),
        right: CipherNode::Leaf(Box::new(CipherLeaf {class_leaf: Vec::new()})),
    });
    match &node.left {
        CipherNode::Leaf(_) => (),
        CipherNode::Internal(left) => flatten_cipher_tree(out, left),
    }
    match &node.right {
        CipherNode::Leaf(_) => (),
        CipherNode::Internal(right) => flatten_cipher_tree(out, right),
    }
}

#[derive(Debug,Clone)] 
pub struct CipherInternal {
    pub threshold_ct: Vec<Vec<RLWECiphertext>>,
    pub index: usize, 
    pub op: Op,
    pub left: CipherNode,
    pub right: CipherNode,
}

#[derive(Debug,Clone)] 
pub struct CipherLeaf{
    pub class_leaf: Vec<RGSWCiphertext>,
}


pub fn encrypt_threshold(sk: &RLWESecretKey, t: usize, ctx: &mut Context) -> Vec<RLWECiphertext>{
    let mut pt = PlaintextList::allocate(Scalar::zero(), ctx.plaintext_count());
    let logn = log2(ctx.poly_size.0);
    let mut out = Vec::new();
    for level in 1..=ctx.level_count.0 {
        assert!(t < ctx.poly_size.0);
        let shift: usize = (Scalar::BITS as usize) - ctx.base_log.0 * level - logn;
        pt.as_mut_tensor().fill_with_element(Scalar::zero());
        *pt.as_mut_polynomial().get_mut_monomial(MonomialDegree(t)).get_mut_coefficient() = Scalar::one() << shift;
        let mut ct = RLWECiphertext::allocate(ctx.poly_size);
        sk.encrypt_rlwe(&mut ct, &pt, ctx.std, &mut ctx.encryption_generator);
        out.push(ct);
    }
    out
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
            if i == index {
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


/// An encrypted node.
pub enum EncNode {
    Internal(Box<EncInternal>),
    Leaf(Box<EncLeaf>),
}

#[derive(Debug,Clone)] 
pub struct EncLeaf{
    pub class_leaf: Vec<RGSWCiphertext>,
}
impl EncNode {
    /// Create a new root from  a plaintext root and encrypted choice bits.
    pub fn new(cipher_root: &CipherNode, rgsw_cts: &mut impl Iterator<Item=RGSWCiphertext>) -> EncNode {
        let ct = rgsw_cts.next().unwrap();
        let mut out = EncInternal {
            ct,
            left: EncNode::Leaf(Box::new(EncLeaf {class_leaf: Vec::new()})),
            right: EncNode::Leaf(Box::new(EncLeaf {class_leaf: Vec::new()})),
        };
        match cipher_root {
            CipherNode::Internal(inner) => new_enc_node(&mut out, inner, rgsw_cts),
            CipherNode::Leaf(_) => panic!("this is a leaf"),
        }
        EncNode::Internal(Box::new(out))
    }

    /// Evaluate the tree.
    pub fn eval_inverse(&self, ctx: &Context, buffers: &mut FourierBuffers<Scalar>) -> Vec<RLWECiphertext> {
        let max_leaf_bits = self.max_leaf_len() as usize;
        let mut out = vec![RLWECiphertext::allocate(ctx.poly_size); max_leaf_bits];
        let mut c = RLWECiphertext::allocate(ctx.poly_size);
        *c.get_mut_body().as_mut_tensor().first_mut() = Scalar::one();
        binary_encode(c.get_mut_body().as_mut_tensor().first_mut());
        eval_enc_node_inverse(&mut out, self, c, ctx, buffers);
        out
    }
    /// Evaluate the tree.
    pub fn eval(&self, ctx: &Context, buffers: &mut FourierBuffers<Scalar>) -> Vec<RLWECiphertext> {
        let max_leaf_bits = self.max_leaf_len() as usize;
        //println!("max_leaf_bits= {}",max_leaf_bits);
        let mut out = vec![RLWECiphertext::allocate(ctx.poly_size); max_leaf_bits];
        let mut c = RLWECiphertext::allocate(ctx.poly_size);
        *c.get_mut_body().as_mut_tensor().first_mut() = Scalar::one();
        binary_encode(c.get_mut_body().as_mut_tensor().first_mut());
        eval_enc_node(&mut out, self, c, ctx, buffers);
        out
    }

    pub fn max_leaf_len(&self) -> usize {
        match self {
            EncNode::Internal(internal) => {
                let l = internal.left.max_leaf_len();
                let r = internal.right.max_leaf_len();
                if l > r {
                    l
                } else {
                    r
                }
            }
            EncNode::Leaf(x) => x.class_leaf.len()
        }
    }
}

/// An encrypted internal node where the ciphertext is the choice bit.
pub struct EncInternal {
    pub ct: RGSWCiphertext,
    pub left: EncNode,
    pub right: EncNode,
}

fn new_enc_node(enc_node: &mut EncInternal, cipher_node: &CipherInternal, rgsw_cts: &mut impl Iterator<Item=RGSWCiphertext>) {
    match &cipher_node.left {
        CipherNode::Leaf(x) => enc_node.left = EncNode::Leaf(Box::new(EncLeaf {class_leaf:x.class_leaf.clone()})),
        CipherNode::Internal(left) => match rgsw_cts.next() {
            None => panic!("missing RGSW ciphertext"),
            Some(ct) => {
                let mut new_node = EncInternal {
                    ct,
                    left: EncNode::Leaf(Box::new(EncLeaf {class_leaf: Vec::new()})),
                    right: EncNode::Leaf(Box::new(EncLeaf {class_leaf: Vec::new()})),
                };
                new_enc_node(&mut new_node, left, rgsw_cts);
                enc_node.left = EncNode::Internal(Box::new(new_node));
            }
        },
    }
    match &cipher_node.right {
        CipherNode::Leaf(x) => enc_node.right = EncNode::Leaf(Box::new(EncLeaf {class_leaf:x.class_leaf.clone()})),
        CipherNode::Internal(right) => match rgsw_cts.next() {
            None => panic!("missing RGSW ciphertext"),
            Some(ct) => {
                let mut new_node = EncInternal {
                    ct,
                    left: EncNode::Leaf(Box::new(EncLeaf {class_leaf: Vec::new()})),
                    right: EncNode::Leaf(Box::new(EncLeaf {class_leaf: Vec::new()})),
                };
                new_enc_node(&mut new_node, right, rgsw_cts);
                enc_node.right = EncNode::Internal(Box::new(new_node));
            }
        },
    }
}

fn eval_enc_node_inverse(out: &mut Vec<RLWECiphertext>, node: &EncNode, b: RLWECiphertext, ctx: &Context, buffers: &mut FourierBuffers<Scalar>) {
    match node {
        EncNode::Leaf(x) => {

            for (bit, ct) in x.class_leaf.iter().zip(out.iter_mut()) {
                let mut tmp = RLWECiphertext::allocate(ctx.poly_size);
                bit.external_product_with_buf(&mut tmp, &b, buffers);
                    ct.update_with_add(&tmp);
                
            }
        }
        EncNode::Internal(node) => {
            let mut left = RLWECiphertext::allocate(ctx.poly_size);
            node.ct.external_product_with_buf(&mut left, &b, buffers);
            let mut right = b;
            right.update_with_sub(&left);

            eval_enc_node_inverse(out, &node.left, left, ctx, buffers);
            eval_enc_node_inverse(out, &node.right, right, ctx, buffers);
        }
    }
}

fn eval_enc_node(out: &mut Vec<RLWECiphertext>, node: &EncNode, b: RLWECiphertext, ctx: &Context, buffers: &mut FourierBuffers<Scalar>) {
    match node {
        EncNode::Leaf(x) => {
            for (bit, ct) in x.class_leaf.iter().zip(out.iter_mut()) {
                let mut tmp = RLWECiphertext::allocate(ctx.poly_size);
                bit.external_product_with_buf(&mut tmp, &b, buffers);
                ct.update_with_add(&tmp);
                
            }
        }
        EncNode::Internal(node) => {
            let mut right = RLWECiphertext::allocate(ctx.poly_size);
            node.ct.external_product_with_buf(&mut right, &b, buffers);
            let mut left = b;
            left.update_with_sub(&right);
            
            eval_enc_node(out, &node.left, left, ctx, buffers);
            eval_enc_node(out, &node.right, right, ctx, buffers);
        }
    }
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


pub fn cipher_tree(dir_data:String,input_size:usize){

    let setup_instant_global = Instant::now(); 
    println!("******************************* step 1: server begin *******************************");
    
    let setup_instant = Instant::now();
    let dir_tree = dir_data;
    let base_path_tree = Path::new(& dir_tree);
    let model_path = base_path_tree.join("model.json");
    let model_f = fs::File::open(model_path).unwrap();
    let root: Node = serde_json::from_reader(model_f).expect("cannot parse json");
    let max_feature_index_add_one = root.max_feature_index_add_one();
    println!("max_feature_index_add_one = {}",max_feature_index_add_one);

    println!("server load decision tree time : {:?}", setup_instant.elapsed());

    println!("server print network");
    root.print_tree(0);

    println!("server generater the parameters..");
    let setup_instant = Instant::now();
    let mut ctx_server = Context::default();
    let sk = ctx_server.gen_rlwe_sk();
    let neg_sk_ct = sk.neg_gsw(&mut ctx_server);//RGSW(-s)
    

    let mut buffers_server = ctx_server.gen_fourier_buffers();
    let ksk_map = gen_all_subs_ksk_fourier(&sk, &mut ctx_server, &mut buffers_server);
    println!( "server generate sk, RGSW(-s), ksk_map time : {:?}",setup_instant.elapsed());

    println!("server encrypt the decision tree");
    let leaf_len = root.max_leaf_len();
    
    println!("leaf_len = {}",leaf_len);
    let cipher_root=CipherNode::enc_tree(&root, &sk, &mut ctx_server, max_feature_index_add_one,leaf_len);
 
    cipher_root.print_cipher_tree(0);
    println!("server encrypt the network time : {:?}", setup_instant.elapsed());
    println!("******************************* step 1: server end *******************************");
    
    println!("******************************* step 2: client begin *******************************");
    println!("client generate the parameters..");
    let setup_instant = Instant::now();
    let ctx_client = Context::default();
    let mut buffers_client = ctx_client.gen_fourier_buffers();

    println!("client generate the parameters time : {:?}", setup_instant.elapsed());
    let setup_instant = Instant::now();

    println!("client load private data");

    //let dir_data = dir_tree;
    //let input_size = 10;
    let base_path = Path::new(& dir_tree);
    let x_test_path = base_path.join("x_test.csv");
    let x_test = parse_csv(&x_test_path);
    println!("client print data");
    for i in 0..input_size{println!("Print x_test[{}]: {:?}",i,x_test[i]);}


    let x_test_input: Vec<Vec<usize>> = x_test.into_iter().take(input_size).collect();

    println!("client load data time : {:?}", setup_instant.elapsed());

    println!("client start to private transform..");
    let setup_instant = Instant::now();
    let flat_cipher_nodes = cipher_root.flatten();

    let server_f = |ct, buffers_client: &mut FourierBuffers<Scalar>| {
        let enc_root = {
            let mut rgsw_cts = compare_expand(&flat_cipher_nodes, ct, &neg_sk_ct, &ksk_map, &ctx_client, buffers_client);
            EncNode::new(&cipher_root, &mut rgsw_cts)
        };
        let final_label_ct = enc_root.eval(&ctx_client, buffers_client);
        final_label_ct
    };

    let output_cts: Vec<Vec<RLWECiphertext>> = x_test_input.iter().map(|ct| {
        println!("server_f");
        server_f(ct, &mut buffers_client)
    }).collect();

    println!( "client evaluate {} line data, finish step 2 time {:?} ", input_size, setup_instant.elapsed());
    println!( "client evalutae {} line data, average finish step 2 time {:?} ", input_size, setup_instant.elapsed()/input_size.try_into().unwrap());
    let setup_instant = Instant::now();
    println!( "client send output");
    println!("*******************************step 2: client end*******************************");
    

    println!("*******************************step 3: server begin *******************************");
    println!("server receive the client's output");

    println!("server decrypt the output");


    for (ct, feature) in output_cts.iter().zip(x_test_input.iter()) {
        let actual_scalar = decrypt_and_recompose(&sk, ct, &ctx_client);
        let expected_scalar = root.eval(feature) as Scalar;
        print!(  " actual_scalar: {:?} ",actual_scalar);
        println!(" expected_scalar   : {:?}",expected_scalar);
        assert_eq!(expected_scalar, actual_scalar);
    }
    println!("server decrypt the output time {:?} \n", setup_instant.elapsed());
    println!("overall time: {:?}", setup_instant_global.elapsed());
}
