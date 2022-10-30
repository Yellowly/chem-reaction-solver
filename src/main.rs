use std::{ops, fmt::Display, cmp, io, str::{Chars, FromStr}, string::ParseError};
use yew::prelude::*;
use web_sys::HtmlInputElement;

fn main() {
    println!("Hello, world!");
    /*
    let test: MeasuredValue = MeasuredValue::from("530 m/s");
    let test2: MeasuredValue = MeasuredValue::from("4390 m/s2");
    */
    let table: PeriodicTable = PeriodicTable::build();

    
    let mut run: bool = true;
    while run {
        let mut input = String::new();
        io::stdin().read_line(&mut input).expect("error");
        input = input.trim().to_string();
        if input.starts_with("parse"){
          println!("{:?}",MeasuredValue::from(input.split_once("parse").unwrap().1))
        }else if input.starts_with("convert2"){
          let mut f = input.split(" ");
          f.next();
          let mut v = f.next().unwrap().parse::<MeasuredValue>().unwrap();
          let u2 = f.next().unwrap().parse::<MeasuredValue>().unwrap();
          print!("{:?} = ",v.to_string());
          v.convert(&u2.unit);
          println!("{:?}",v);
        }else if input.starts_with("convert"){
          let mut f = input.split(" ");
          f.next();
          let v = f.next().unwrap().parse::<f32>().unwrap();
          let u1 = f.next().unwrap().parse::<Unit>().unwrap();
          let u2 = f.next().unwrap().parse::<Unit>().unwrap();
          println!("{}{} = {}{}",v,u1.to_string(),v*u1.conversion_factor(&u2),u2.to_string());
        }else if input.starts_with("formula"){
          let f = Formula::make(input.split_once(" ").unwrap().1);
          let mut vals: Vec<String> = Vec::new();
          for i in 0..f.len(){
            input = String::new();
            io::stdin().read_line(&mut input).expect("error");
            vals.push(input.trim().to_string());
          }
          let res = f.solve(vals);
          println!("{:?} => {:?}",res.to_string(),res.show());
        }else if input.contains("=") {
          let solution = MeasuredValue::equat(&input);
          println!("{:?} = {:?}",solution.to_string(),solution.show());
          let parser = Tokenizer::make(vec![Token::MesVal,Token::Op]);
          let parsed = parser.parse(&input);
          println!("{:?}",parsed);
        } else if input.eq("end") {
            break
        } else {
          let solution = MeasuredValue::eval(&input);
          println!("{:?} = {:?}",solution.to_string(),solution.show());
          let parser = Tokenizer::make(vec![Token::MesVal,Token::Op]);
          let parsed = parser.parse(&input);
          println!("{:?}",parsed);
        }
    }
    println!("ended!");
    yew::start_app::<CounterComponent>();

    /*

    println!("{:?}",test);
    println!("{:?}",test2);
    println!("{:?}",(test/test2).to_string());
    println!("{:?}",solveVolume(mvf("760 mmHg"),mvf("1 mol"),mvf("293.15 K"),solveConstant(mvf("742.46 mmHg"),mvf("2.97 L"),mvf("1 mol"),mvf("273.15 K"))));
    println!("{:?}",solveVolume(mvf("760 mmHg"),mvf("0.132 mol"),mvf("293.15 K"),mvf("62.36")));

    println!("t1 {:?}",mvf("720 mmHg")*mvf("0.038 L")*mvf("273.15 K")/(mvf("760 mmHg")*mvf("300.15 K")));

    println!("t2 {:?}",mvffn!("720 mmHg"*"0.038 L"*"273.15 K"/"760 mmHg"/"300.15 K"));

    println!("t3 {:?}",MeasuredValue::eval("720 mmHg*0.038 L*273.15 K/(760 mmHg*300.15 K)").to_string());

    println!("test rounder {:?}",keep_digits(0.00159265,5));

    println!("test rounder 2 {:?}",keep_digits(0.120000005,3));
    println!("eval test: {:?}",MeasuredValue::eval("3m/5s"));

    println!("measured value equat! test {:?}",equat!(MeasuredValue, MeasuredValue{value:1.0,unit:Vec::new(),sigfigs: u32::MAX},"Q"+"-75 C"*"140 g"*"4.182 J/g*C"+"20 C"*"140 g"*"4.182 J/g*C"));

    println!("eval equat test: {:?}",MeasuredValue::eval("140g * 4.182 J/g*C * (75 C - 20 C)"));

    println!("eval sub test: {:?}",MeasuredValue::eval("-(5m/1s)"));

    println!("equat solve: {:?}",MeasuredValue::equat("32201.398 J = 140 g * 4.182 J/g*C * (75 C - ?)"));

    println!("{:?}",get_base_unit_varient("g"));

    println!("{:?}",equat!(f32,1.0,"1"/"x"+"-5"));

    println!("{:?}",equat!(f32,1.0,"x"/"2"+"-5"));

    println!("{:?}",equat!(f32,1.0,"Q"+"-75"*"140"*"4.182"+"20"*"140"*"4.182"));

    println!("{:?}",MeasuredValue::default()-mvf("720 mmHg"))
*/
    //println!("{:?}",test);

    //println!("{:?}",MeasuredValue::eval("340.0g * -334 J/g"));

    //yew::start_app::<CounterComponent>();
    //println!("{:?}",keep_digits(4.184*375.0,3));
    //println!("{:?}",MeasuredValue::equat("375g*4.184J/g*C*(?-32C)=125g*4.184J/g*C*(?-92C)"));

    //println!("{:?}",table.elem_from_symbol("Mg").to_moles(mvf("3.20g")));

    /* 
    let mut arr : Vec<Vec<f32>> = vec![vec![-18.0,0.0,2.0,0.0,0.0],vec![-8.0,0.0,0.0,1.0,0.0],vec![0.0,-2.0,1.0,2.0,0.0]];
    println!("{:?}",arr);
    //println!("{}",findGCF(&[12,18,24]));
    
    arr = to_reduced_roechelon(arr);
    println!("{:?}",arr);*/

    //swap(arr);

    //test STUFF!!!
    //75g*4.184J/g*C*(19C-35C)=-1*?*4.184J/g*C*(19C-10.0C) but without eq aka 75g*4.184J/g*C*(19C-35C)+?*4.184J/g*C(19C-10.0C)=0.000
    //228 J = 125 g * ? * (27.12 C - 22.38 C) check sig figs
  //(4.000*14.7g)*4.184J/g*C*(?-0.00C)=-250g*4.184J/g*C*(?-61.1C) = 58.8g*4.184J/g*C*(?-0.00C)=-250g*4.184J/g*C*(?-61.1C)
  //(-411.0 * 241.8) -( -426.7 + -92.3 )
  //(-4.02 mol_H2O(H2)(l)+5.30 mol_H2O(H2)(l))
}

#[macro_export]
macro_rules! testthing(
    ($($op:expr)*) => {
      $(
        &op
      )*
    }
);



#[macro_export]
macro_rules! mvffn(
    ($($v:literal $($op:tt)? )*) => {
      $(
        MeasuredValue::from($v) $($op)?
      )*
    }
);

#[macro_export]
macro_rules! test(
    ($($v:literal $($op:tt)?)*) => {
      $(
        $v.parse::<f32>().unwrap_or(1.0) $($op)?
      )*
    }
);


#[macro_export]
macro_rules! multi_matcher{
    ($inputname:ident {$( $mat:expr, ($($alias:expr),*) ),*, _ $default:expr } ) =>{
        match($inputname){$(
            $(
                $alias => $mat,
            )*
        )*
            _ => $default,
        }
    }
}

#[macro_export]
macro_rules! equat(
    ($t:ty, $one:expr, $($v:literal $($op:tt)?)*) => {
        ($(
            $v.parse::<FormulaVar<$t>>().unwrap_or(FormulaVar{varcoef:$one,val: <$t>::default(), inv:1}) $($op)?
        )*).solve()
    }
);

#[derive(Debug)]
struct FormulaVar<T>{
  varcoef: T,
  val: T,
  inv: i8,
}

impl<T> std::str::FromStr for FormulaVar<T> where T: std::str::FromStr, T:std::default::Default{
  type Err = f32;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let n = s.parse::<T>();
        match n{
            Ok(v) => Ok(FormulaVar { varcoef: T::default(), val: v, inv:1}),
            Err(_y) => Err(-1.0)
        }
    }
}

impl<T> FormulaVar<T> where T: std::ops::Div<Output = T>,T: std::ops::Mul<f32,Output = T>, T: std::fmt::Debug{
    fn solve(self) -> T{
        if self.inv<0 {return self.varcoef/(self.val*-1.0)}
        (self.val*-1.0)/self.varcoef
    }
}
        
impl<T> std::ops::Add<FormulaVar<T>> for FormulaVar<T> where T: std::ops::Add<Output = T>{
    type Output = FormulaVar<T>;
    fn add(self, _rhs: FormulaVar<T>) -> FormulaVar<T>{
        FormulaVar{varcoef: self.varcoef+_rhs.varcoef, val: self.val+_rhs.val, inv:self.inv}
    }
}

impl<T> std::ops::Sub<FormulaVar<T>> for FormulaVar<T> where T: std::ops::Sub<Output = T>{
    type Output = FormulaVar<T>;
    fn sub(self, _rhs: FormulaVar<T>) -> FormulaVar<T>{
        FormulaVar{varcoef: self.varcoef-_rhs.varcoef, val: self.val-_rhs.val, inv:self.inv}
    }
}

impl<T> std::ops::Mul<FormulaVar<T>> for FormulaVar<T> where T: std::ops::Mul<Output = T>, T: std::ops::Add<Output = T>, T: std::clone::Clone{
    type Output = FormulaVar<T>;
    fn mul(self, _rhs: FormulaVar<T>) -> FormulaVar<T>{
        FormulaVar{varcoef: self.varcoef*_rhs.val.clone()+_rhs.varcoef*self.val.clone(), val: self.val*_rhs.val, inv:self.inv}
    }
}

impl<T> std::ops::Div<FormulaVar<T>> for FormulaVar<T> where T: std::ops::Div<Output = T>, T: std::ops::Mul<Output = T>, T: std::ops::Add<Output = T>, T: std::clone::Clone, T: std::cmp::PartialEq, T:std::default::Default {
    type Output = FormulaVar<T>;
    fn div(self, _rhs: FormulaVar<T>) -> FormulaVar<T>{
        if _rhs.varcoef!=T::default() {
            if _rhs.val==T::default() {return FormulaVar{varcoef: self.val.clone()/_rhs.varcoef, val: T::default(), inv: -1*self.inv}}
            return FormulaVar{varcoef: self.val.clone()/_rhs.varcoef, val: self.val/(_rhs.val), inv: -1*self.inv}
        }
        FormulaVar{varcoef: self.varcoef/(_rhs.val.clone()), val: self.val/(_rhs.val), inv: self.inv}
    }
}






fn get_unit_varients(units: Vec<(String,i8)>){

}



//copy pasted off the wikipedia article for reduced row echelon form
fn to_reduced_roechelon(mut m: Vec<Vec<f32>>) -> Vec<Vec<f32>>{
    let mut lead: usize = 0;
    let row_count: usize = m.len();
    let column_count : usize = m[0].len();
    for r in 0..row_count{
        if column_count <= lead {
            return m //out of bounds
        }
        let mut i: usize = r;
        while m[i][lead] == 0.0{
            i+=1;
            if row_count == i{
                i=r;
                lead+=1;
                if column_count == lead{
                    return m//lead has gone through the whole matrix
                }
            }
        }
        if i != r {
            //swap rows i and r
            m.swap(i,r);
        }
        //divide row r by m[r, lead]
        let thing = m[r][lead];
        for n in 0..m[r].len(){
            m[r][n]/=thing;
        }
        for i in 0..row_count{
            if i != r {
                //Subtract M[i, lead] multiplied by row r from row i
                let temp = m[i][lead];
                for j in 0..column_count{
                    m[i][j]-=temp*m[r][j];
                }
            }
        }
    lead+=1_usize;
    }
    m
}


fn gcf(a: i32, b: i32) -> i32{
    if a == 0{
        return b
    }
    gcf(b % a, a)
}

fn findGCF(nums: &[i32]) -> i32{
    let mut res: i32 = nums[0];
    for n in nums{
        res = gcf(res,*n);
        if res == 1{
            return 1
        }
    }
    res
}
//binary magic
fn make_op(c: char, p: u8) -> u16{
    //alternatively (p*3) << 4
    (p as u16) *3*16 + match c{
        '+' => 1,
        '-' => 2,
        '*' => 19, //3 + 16 = 00010011
        '×' => 19,
        '/' => 20, //4 + 16 = 00010100
        _ => 0
    }
}

fn solvePressure(volume: MeasuredValue, num_particles: MeasuredValue, temp: MeasuredValue, r_constant: MeasuredValue) -> MeasuredValue{
    (num_particles*r_constant*temp)/volume
}
fn solveVolume(pressure: MeasuredValue, num_particles: MeasuredValue, temp: MeasuredValue, r_constant: MeasuredValue) -> MeasuredValue{
    (num_particles*r_constant*temp)/pressure
}
fn solveTemp(pressure: MeasuredValue, volume: MeasuredValue, num_particles: MeasuredValue, r_constant: MeasuredValue) -> MeasuredValue{
    (pressure*volume)/(num_particles*r_constant)
}
fn solveNumParticles(pressure: MeasuredValue, volume: MeasuredValue, temp: MeasuredValue, r_constant: MeasuredValue) -> MeasuredValue{
    (pressure*volume)/(temp*r_constant)
}
fn solveConstant(pressure: MeasuredValue, volume: MeasuredValue, num_particles: MeasuredValue, temp: MeasuredValue) -> MeasuredValue{
    (pressure*volume)/(num_particles*temp)
}

fn pressureConverter(value: f32, unit: &str) -> f32{
    value/match unit{
        "kPa"=>101.325,
        "mmHg" => 760.0,
        "Torr" => 760.0,
        "psi" => 14.695,
        "Pa" => 101325.0,
        _ => 1.0,
    }
}

#[derive(Clone,Debug)]
struct Operator{
  name: char,
  priority: u8,
}
impl Tokenizable for Operator{
  type Res = Operator;
  fn extract(idx: usize, raw: &Vec<char>) -> (usize,Option<Self::Res>){
    for i in idx..raw.len(){
      if !raw[i].is_whitespace(){//&& != ')'
        let res = Operator::make(raw[i]);
        return(i,res)
      }
    }
    (raw.len(),Option::None)
    }
    fn set_priority(&mut self,paren_count:u8){
        self.priority+=paren_count*3;
    }
}
impl Operator{
  fn make(c: char) -> Option<Operator>{
    return match c{
      '+' => Option::Some(Operator{name: '+',priority: 0}),
      '-' => Option::Some(Operator{name: '-',priority: 0}),
      '*' => Option::Some(Operator{name: '*',priority: 1}),
      '/' => Option::Some(Operator{name: '/',priority: 1}),
      '×' => Option::Some(Operator{name: '*',priority: 1}),
      '÷' => Option::Some(Operator{name: '/',priority: 1}),
      _ => Option::None
    }
  }
  fn build(c: char, pri: u8) -> Option<Operator>{
    let r: Option<Operator> = Operator::make(c);
    if r.is_none() {return Option::None};
    let mut res = r.unwrap();
    res.priority+=pri;
    Option::Some(res)
  }
  fn is_op(c: char) -> bool{
    return match c{
      '+' => true,
      '-' => true,
      '*' => true,
      '/' => true,
      '×' => true,
      '÷' => true,
      _ => false
    }
  }
}
impl ToString for Operator{
  fn to_string(&self) -> String{
    self.name.to_string()
  }
}
#[derive(Clone,Debug,PartialEq)]
enum Token{
    MesVal,
    Op,
    Value,
    Unit,
    MulUnits,
}
#[derive(Clone,Debug)]
enum TokenVal{
    MesVal(MeasuredValue),
    Op(Operator),
    Value(Value),
    Unit(Unit),
    MulUnits(MulUnits),
    Error(usize), //index of the error
}
impl TokenVal{
  fn make(t: &Token, p: u8, start: usize, s: &Vec<char>) -> (usize, TokenVal){
    match t{
      Token::MesVal => {let r = MeasuredValue::extract(start,s); return match r.1{Some(v) => (r.0,TokenVal::MesVal(v)), None => (r.0,TokenVal::Error(r.0))}},
      Token::Op => {let r = Operator::extract(start,s); return match r.1{Some(mut v) => {v.set_priority(p); (r.0,TokenVal::Op(v))}, None => (r.0,TokenVal::Error(r.0))}},
      Token::Value => {let r = Value::extract(start,s); return match r.1{Some(mut v) => {v.set_priority(p); (r.0,TokenVal::Value(v))}, None => (r.0,TokenVal::Error(r.0))}},
      Token::Unit => {let r = Unit::extract(start,s); return match r.1{Some(mut v) => {v.set_priority(p); (r.0,TokenVal::Unit(v))}, None => (r.0,TokenVal::Error(r.0))}},
      Token::MulUnits => {let r = MulUnits::extract(start,s); return match r.1{Some(mut v) => {v.set_priority(p); (r.0,TokenVal::MulUnits(v))}, None => (r.0,TokenVal::Error(r.0))}},
    }
  }
}
struct Tokenizer{
    tokens: Vec<Token>,
}
impl Tokenizer{
    fn make(tokens: Vec<Token>) -> Tokenizer{
        Tokenizer{tokens}
    }
    fn parse(&self, raw: &str) -> (Vec<TokenVal>,Option<usize>){
        let t: Vec<char> = raw.chars().collect::<Vec<char>>();
        let len = t.len();
        let mut i: usize = 0_usize;
        let mut res: Vec<TokenVal> = Vec::new();
        let mut idx: usize = 0_usize;
        let mut haserr: Option<usize> = Option::None;
        let mut paren_count: u8 = 0;
        while i<len{
            if t[i]=='('{paren_count+=1; i+=1; continue}else if t[i]==')'{paren_count-=1; i+=1; continue}
            if t[i].is_whitespace(){i+=1; continue}
            println!("parse {:?} from {:?}",self.tokens[idx],i);
            let extracted: (usize,TokenVal)=TokenVal::make(&self.tokens[idx],paren_count, i,&t);
            i=extracted.0+1;
            if haserr.is_none(){
                haserr = match extracted.1 {
                TokenVal::Error(place) => Option::Some(place),
                _ => Option::None,
                }
            }
            res.push(extracted.1);
            idx+=1;
            if idx>=self.tokens.len(){idx=0}
        }
        (res,haserr)
    }
}
trait Tokenizable{
  type Res;
  fn extract(idx: usize, raw: &Vec<char>) -> (usize,Option<Self::Res>);
  fn set_priority(&mut self,paren_count:u8){
    
  }
}

trait Solver{
    fn from(s: &str) -> (Vec<TokenVal>,Option<usize>);
    fn solve_with_work(tokens: Vec<TokenVal>) -> (Option<String>,Vec<String>);
    fn solve(tokens: Vec<TokenVal>) -> Option<String>{
        Self::solve_with_work(tokens).0
    }
}

trait Prioritizer{
  type Res;
}


struct Formula{
  formula: String,
  ops: Vec<u16>,
  coefs: Vec<f32>,
  equat: bool,
}
impl Formula{
  fn make(formula: &str) -> Formula{
    let mut equat: bool = false;
    let mut formatted = formula.to_string();
    if formula.contains("="){
      equat = true;
      formatted = formatted.replace("=","-(").to_owned()+")"
    }
    formatted+="+";
    let mut ops: Vec<u16> = Vec::new();
    let mut coefs: Vec<f32> = Vec::new();
    let mut priority: u8 = 0;
    let mut prev: char = ' ';
    let mut incoef: bool = true;
    let mut coefbuilder: String = String::new();
    for c in formatted.chars(){
      if c.is_whitespace() {continue};
      if incoef{
        if c.is_numeric() || c=='.' || c=='-'{
          coefbuilder.push(c);
          prev = c;
          continue;
        }else{
          if coefbuilder.len()==1 && coefbuilder=="-"{coefbuilder.push('1')};
          coefs.push(coefbuilder.parse::<f32>().unwrap_or(1.0));
          coefbuilder=String::new();
          incoef = false;
        }
      }
      if make_op(c,0) & 15 != 0{
        ops.push(make_op(c,priority));
        incoef = true;
      }else if c == '('{
        if make_op(prev,0)==0{
          ops.push(make_op('*',priority));
        }
        priority+=1;
      }else if c==')'{
        priority-=1;
      }
      prev=c;
    }
    println!("coefs: {:?}, ops: ",coefs);
    //assert!(coefs.len()==ops.len()+1);
    Formula{formula: formula.to_string(), ops, coefs, equat}
  }
  fn solve(&self, vals: Vec<String>) -> MeasuredValue{
    //assert!(vals.len()==(&self.ops).len()+1);
    let mut opstack: Vec<u16> = Vec::new();
    if (&self.equat).clone(){
      let mut valstack: Vec<FormulaVar<MeasuredValue>> = Vec::new();
      for i in 0..(&self.ops).len()+(vals.len()){
          if i%2==1{
              let op: u16 = (&self.ops)[i/2];
              while opstack.len()>0 && opstack.last().unwrap().clone() & 65520 >= op & 65520{
                  let o: u16 = opstack.pop().unwrap();
                  let right: FormulaVar<MeasuredValue> = valstack.pop().unwrap();
                  let left: FormulaVar<MeasuredValue> = valstack.pop().unwrap();
                  let val: FormulaVar<MeasuredValue> = match o & 15 {
                  1 => left+right,
                  2 => left-right,
                  3 => left*right,
                  4 => left/right,
                  _ => left+right,
                  };
                  valstack.push(val);
              }
              opstack.push(op);
          }else{
              valstack.push(vals[i/2].parse::<FormulaVar<MeasuredValue>>().unwrap_or(FormulaVar{varcoef: MeasuredValue{value:1.0,unit:MulUnits::default(),sigfigs:u32::MAX-128},val: MeasuredValue::default(),inv:1}));
          }
      }
      return valstack.pop().unwrap().solve()
    }else{
      let mut valstack: Vec<MeasuredValue> = Vec::new();
      for i in 0..(&self.ops).len()+(vals.len()){
          if i%2==1{
              let op: u16 = (&self.ops)[i/2];
              while opstack.len()>0 && opstack.last().unwrap().clone() & 65520 >= op & 65520{
                  let o: u16 = opstack.pop().unwrap();
                  let right: MeasuredValue = valstack.pop().unwrap();
                  let left: MeasuredValue = valstack.pop().unwrap();
                  let val: MeasuredValue = match o & 15 {
                  1 => left+right,
                  2 => left-right,
                  3 => left*right,
                  4 => left/right,
                  _ => left+right,
                  };
                  valstack.push(val);
              }
              opstack.push(op);
          }else{
              valstack.push(vals[i/2].parse::<MeasuredValue>().unwrap_or(MeasuredValue::build(-1.0,1,MulUnits::default_error())));
          }
      }
      return valstack.pop().unwrap()
    }
  }
  fn len(&self) -> usize{
    (&self.coefs).len()
  }
}

fn keep_digits(n: f32, d: u32) -> f32{
    if d>32 {return n}
    let starting_digit: i32 = starting_pow(n);
    let d: i32 = d as i32;
    //i love floating point operations... 
    if d-starting_digit-1 > 0 {(n*10_f32.powf((d-starting_digit-1) as f32)).trunc()/10_f32.powf((d-starting_digit-1) as f32)}
    else {(n*10_f32.powf((d-starting_digit-1) as f32)).trunc()*10_f32.powf((starting_digit-d+1) as f32)}
}
fn starting_pow(n: f32) -> i32{
    let mut temp: f32 = 0.0;
    if n!=0.0 {temp = n.abs().log10()};    
    let mut p: i32 = temp as i32;
    if temp < 0.0{
        p-=1;
    }
    p
}
fn shift_digits(n: f32, d: i32) -> f32{
    if d==0 {n}
    else if d<0 {n/10_f32.powf(-d as f32)}
    else {n*10_f32.powf(d as f32)}
}

fn round_digits(n: f32, d: u32) -> f32{
    if d>32 {return n}
    let starting_digit: i32 = starting_pow(n);
    let d: i32 = d as i32;
    //i love floating point operations... 
    if d-starting_digit-1 > 0 {(n*10_f32.powf((d-starting_digit-1) as f32)).round()/10_f32.powf((d-starting_digit-1) as f32)}
    else {(n*10_f32.powf((d-starting_digit-1) as f32)).round()*10_f32.powf((starting_digit-d+1) as f32)}
}

enum MassUnits{
    mg,
    cg,
    dg,
    g,
    dag,
    hg,
    kg,
}

#[derive(Debug, PartialEq, Clone)]
enum UnitType{
    Volume,
    Area,
    Length,
    Pressure,
    Mass,
    Time,
    Energy,
    Force,
    Temperature,
    None,
}
#[derive(Debug,PartialEq,Clone)]
struct Unit{
  unit: String,
  exp: i8,
  id: String,
  state: String,
  unit_type: UnitType
}
impl Tokenizable for Unit{
  type Res = Unit;
  fn extract(idx: usize, raw: &Vec<char>) -> (usize,Option<Unit>){
    let mut unit: String = String::new();
    let mut exp: String = String::new();
    let mut id: String = String::new();
    let mut state: String = String::new();

    let mut end: usize = idx;
    
    let mut stage: u8 = 0;
    for i in idx..raw.len(){
      let c: char = raw[i];
      println!("{}",c);
      if c.is_whitespace(){
        end=i;
        continue
      }else if Operator::is_op(c) || (stage!=3&&stage!=2) && c==')'{
        end=i-1;
        //if stage==0{return (end,Option::None)}
        break
      }
      match stage{
        0 => {
          if c.is_numeric(){if unit.is_empty() {return (i,Option::None)}; stage+=1; exp.push(c)}
          else if c=='_'{stage=2}
          else if !c.is_alphabetic(){println!("pushed op on {}",c); if unit.is_empty() {return (i,Option::None)}; end=i-1; break}
          else {unit.push(c)}
        }
        1 => {
          if c=='_' {stage+=1}
          else {exp.push(c)}
        }
        2 =>{
          if c=='('{if raw[i+1].is_alphabetic() {if raw[i+1].is_uppercase(){id.push(c)}else{stage+=1}} else {end=i-1; break}}
          else {id.push(c)}
        }
        _ =>{
          if c==')' {end=i; break}
          else {state.push(c)}
        }
      }
      end=i;
    }
    if unit.is_empty(){return (end,Option::None)}
    let mut res: Unit = Unit{unit,exp: exp.parse::<i8>().unwrap_or(1),id,state,unit_type: UnitType::None};
    res.unit_type = res.get_base_unit_varient();
    (end,Option::Some(res))
  }
}
impl Unit{
  fn eq_unit(&self, other: &Self) -> bool{
    return self.unit == other.unit;
  }
  fn eq_except_exp(&self, other: &Self) -> bool{
      return self.unit==other.unit && self.id==other.id && self.state==other.state;
  }
  //given metric prefix, matches the conversion factor
  fn metric_convert(&self) -> f32{
    let unit = &self.unit;
    if unit.len()<2 {return 1.0};
    if &unit[0..2]=="da" {return 10.0};
    match &unit[0..1]{
      "T" => 1000000000000.0,
      "G" => 1000000000.0,
      "M" => 100000.0,
      "k" => 1000.0,
      "h" => 100.0,
      "d" => 0.1,
      "c" => 0.01,
      "m" => 0.001,
      "µ" => 0.000001,
      "n" => 0.000000001,
      "p" => 0.000000000001,
      _ => 1.0
    }
  }
  fn is_metric(&self) -> bool{
      let unit = &self.unit;
      if unit.is_empty(){return false}
      matches!(&unit[unit.len()-1..unit.len()],"J"|"L"|"g"|"m"|"N")||unit.ends_with("Pa")
  }
  
  fn conversion_factor(&self, to: &Unit) -> f32{
      let to_ext: &str = &to.unit;
      let from_ext: &str = &self.unit;
      if self.is_metric(){
          if to.is_metric(){
              return self.metric_convert()/to.metric_convert()
          }
          return self.metric_convert()/match to_ext{
              "mol" => {let tid = &to.id; let fid = &self.id; if tid!=fid{if tid.is_empty(){Element::molar_mass(&fid)}else if fid.is_empty(){Element::molar_mass(&tid)}else{-0.0}}else{Element::molar_mass(&tid)}},
              "mole" => {let tid = &to.id; let fid = &self.id; if tid!=fid{if tid.is_empty(){Element::molar_mass(&fid)}else if fid.is_empty(){Element::molar_mass(&tid)}else{-0.0}}else{Element::molar_mass(&tid)}},
              _ => 1.0
          }
      }
      if to.is_metric(){
          return match from_ext{
              "mol" => {let tid = &to.id; let fid = &self.id; if tid!=fid{if tid.is_empty(){Element::molar_mass(&fid)}else if fid.is_empty(){Element::molar_mass(&tid)}else{-0.0}}else{Element::molar_mass(&tid)}},
              "mole" => {let tid = &to.id; let fid = &self.id; if tid!=fid{if tid.is_empty(){Element::molar_mass(&fid)}else if fid.is_empty(){Element::molar_mass(&tid)}else{-0.0}}else{Element::molar_mass(&tid)}},
              _ => 1.0
          }/to.metric_convert()
      }
      0.0
  }
  fn convert_val(&self, val: f32, to: &Unit) -> f32{
      let to_ext: &str = &to.unit;
      let from_ext: &str = &self.unit;
      let mut val = val;
      if to_ext=="K"&&from_ext=="C"{
          val+=273.15;
      }else if to_ext=="C"&&from_ext=="K"{
          val-=273.15;
      }else{
          if val==0.0 {return 0.0}
          val*=self.conversion_factor(to).powi(self.exp as i32);
      }
      val
  }
  /*
  fn extract_unit(unit: &str) -> String{
      match unit.split_once('_'){
          Some(u) => u.0.to_string(),
          None => unit.to_string()
      }
  }
  fn extract_id(unit: &str) -> String{
      match unit.split_once('_'){
          Some(u) => u.1.to_string(),
          None => String::from(""),
      }
  }*/
  fn get_base_unit_varient(&self) -> UnitType{
    if self.is_metric(){
      if self.unit.ends_with("Pa") {return UnitType::Pressure}
      return match &self.unit[self.unit.len()-1..self.unit.len()]{
        "J" => UnitType::Energy,
        "m" => UnitType::Length,
        "L" => UnitType::Volume,
        "g" => UnitType::Mass,
        "N" => UnitType::Force,
        "C" => UnitType::Temperature,
        _ => UnitType::None,
      }
    }
    //NEEDS IMPERIAL UNITS
    if self.unit.ends_with("mol"){return UnitType::Mass};
    UnitType::None
  }
  fn get_unit_varient(&self) -> (UnitType,i8){
    (self.unit_type.clone(),self.exp)
  }
  fn same_base_type(&self, other: &Self) -> bool{
    self.unit_type==other.unit_type
  }
  fn same_type(&self, other: &Self) -> bool{
    self.same_base_type(other) && self.exp==other.exp
  }
  fn mul_pow(&mut self,m: i8){
    self.exp*=m;
  }
}
impl PartialEq<str> for Unit{
    fn eq(&self, rhs: &str) -> bool{
        (&self.unit)==rhs
    }
}
impl ToString for Unit{
    fn to_string(&self) -> String{
        let mut res: String = String::new();
        format!("{}{}",self.unit,self.exp.to_string());
        if !self.id.is_empty(){
            res.push('_');
            res.push_str(&self.id.to_string());
            if !self.state.is_empty(){
                res.push_str(&format!("({})",self.state));
            }
        }
        res
    }
}
impl FromStr for Unit{
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let r = Unit::extract(0, &s.chars().collect::<Vec<char>>());
        match r.1{
            Some(v) => return Result::Ok(v),
            None => return Result::Err(format!("Parsing error at idx {}",r.0))
        }
    }
}

#[derive(Clone,Default,Debug)]
struct MulUnits{
  units: Vec<Unit>
}
impl Tokenizable for MulUnits{
  type Res = MulUnits;
  fn extract(idx: usize, raw: &Vec<char>) -> (usize,Option<Self::Res>){
    let mut end: usize = idx;
    let mut prev_end: usize = end;
    if prev_end>0 {prev_end=end-1};
    let mut units: Vec<Unit> = Vec::new();
    let mut exp_mul: i8 = 1;
    while end < raw.len(){
      println!("extrating unit from: {}",end);
      let u = Unit::extract(end,raw);
      //println!("extract unit end {:?}",u);
      end=u.0;
      if u.1.is_none(){
        //println!("returning {:?}", prev_end);
        end=prev_end;
        break
      }
      let mut unit: Unit = u.1.unwrap();
      unit.mul_pow(exp_mul);
      let search = units.iter().position(|un| unit.eq_except_exp(un));
      match search{
          Some(v) => units[v].exp+=1,
          None => units.push(unit)
      }
      let next_op = Operator::extract(end+1,raw);
      println!("next op in unit: {:?}",next_op);
      if next_op.1.is_some(){
        let unwrapped = next_op.1.unwrap();
        match unwrapped.name{
            '*' => {prev_end=end; end = next_op.0+1},
            '/' => {prev_end=end; end = next_op.0+1; exp_mul*=-1;},
            _ => {break}
        }
      }else {break}
    }
    (end,Option::Some(MulUnits{units}))
  }
}


impl PartialEq for MulUnits{
  fn eq(&self, other: &Self) -> bool{
    for u in &self.units{
      if !other.units.contains(&u){
        return false
      }
    }
    return true
  }
}
impl MulUnits{
  fn same_base_types(&self, other: &Self) -> bool{
    for u in &self.units{
      if !other.units.iter().any(|u2| u.same_base_type(u2)){
        return false
      }
    }
    true
  }
  fn same_varients(&self, varients: Vec<(UnitType,i8)>) -> bool{
    for u in &self.units{
        if !varients.iter().any(|u2| u.unit_type==u2.0 && u.exp==u2.1){
            return false
        }
    }
    true
  }
  fn convert_val(&mut self, val: f32, other: &Self) -> f32{
      let mut val = val;
      for (i,u1) in self.get_varients().iter().enumerate(){
        for (j,u2) in other.get_varients().iter().enumerate(){
            if u1.0==u2.0{
                val=self.units[i].convert_val(val,&other.units[j]);
                self.units[i].unit=other.units[j].unit.clone();
            }
        }
      }
      val
  }
  fn get_varients(&self) -> Vec<(UnitType,i8)>{
    let mut res: Vec<(UnitType,i8)> = Vec::new();
    for u in self.units.iter(){
      res.push((u.unit_type.clone(),u.exp));
    }
    res
  }
  fn is_empty(&self) -> bool{
      return self.units.is_empty();
  }
  fn len(&self) -> usize{
      return self.units.len();
  }
  fn default_error() -> MulUnits{
      MulUnits{units: vec![Unit::from_str("Error").unwrap()]}
  }
}
impl FromStr for MulUnits{
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let r = MulUnits::extract(0, &s.chars().collect::<Vec<char>>());
        match r.1{
            Some(v) => return Result::Ok(v),
            None => return Result::Err(format!("Parsing error at idx {}",r.0))
        }
    }
}
impl ToString for MulUnits{
    fn to_string(&self) -> String {
        let mut res: String = String::new();
        let mut numerator: String = String::new();
        let mut denominator: String = String::new();
        for u in &self.units{
            if u.exp<0 {if denominator.len()>0 {denominator.push('*')}; denominator.push_str(&(u.unit)); if u.exp.abs()!=1 {denominator.push_str(&u.exp.abs().to_string())}}
            else {if numerator.len()>0 {numerator.push('*')}; numerator.push_str(&(u.unit)); if u.exp!=1 {numerator.push_str(&u.exp.abs().to_string())}}
        }
        if numerator.is_empty(){
            numerator+="1";
        }else{
            //res.push_str("(");
            res.push_str(&numerator);
            //res.push_str(")");
        }
        if denominator.len()>0{
            res.push_str("/");//(
            res.push_str(&denominator);
            //res.push(')');
        }
        res
    }
}
impl ops::Mul<MulUnits> for MulUnits{
    type Output = MulUnits;

    fn mul(self, rhs: MulUnits) -> Self::Output {
        let mut res = self.clone();
        for u in rhs.units.iter(){
            let index_of = res.units.iter().position(|un| u.eq_except_exp(un));
            match index_of{
                Some(idx) => {
                  if res.units[idx].exp+u.exp==0 {let _ = res.units.swap_remove(idx);}
                  else{res.units[idx].exp+=u.exp}
                },
                None => res.units.push(u.clone())
            }
        }
        res
    }
}
impl ops::Div<MulUnits> for MulUnits{
    type Output = MulUnits;

    fn div(self, rhs: MulUnits) -> Self::Output {
        let mut res = self.clone();
        for u in rhs.units.iter(){
            let index_of = res.units.iter().position(|un| u.eq_except_exp(un));
            match index_of{
                Some(idx) => {
                  if res.units[idx].exp==u.exp {let _ = res.units.swap_remove(idx);}
                  else{res.units[idx].exp-=u.exp}
                },
                None => {let mut pushed = u.clone(); pushed.mul_pow(-1); res.units.push(pushed)}
            }
        }
        res
    }
}





#[derive(Clone,Debug,Default)]
struct Value{
    val: f32,
    sigfigs: u32,
}
impl Tokenizable for Value{
    type Res = Value;
    fn extract(idx: usize, raw: &Vec<char>) -> (usize,Option<Self::Res>){
        if raw.len() - idx == 0{
            return (idx+1,Option::None)
        }
        let mut float_val: String = String::new();
        let mut setfigs: String = String::new();
        let mut exp: String = String::new();
        let mut sigfigs: u32 = 0;
        let mut zero_counter: u32 = 0;
        let mut end: usize = idx;
        //0 = normal number, 1 = past decimal place, 2 = sci notation, 3 = set figs
        let mut stage = 0;
        for i in idx..raw.len(){
            let c = raw[i];
            if c.is_whitespace() {continue};
            match stage{
                0 => {
                    if c.is_numeric() || c=='-'{
                        float_val.push(c);
                        if c=='0' {zero_counter+=1} else if c!='-'{if sigfigs==0 && zero_counter>0 {zero_counter=0}; sigfigs+=1+zero_counter; zero_counter=0};
                    }
                    else if float_val.is_empty() {return (i,Option::None)}
                    else if c=='.' {float_val.push('.'); stage=1;}
                    else if c=='E' {stage=2;}
                    else if c=='~' {stage=3;}
                    else {end=i-1; break}
                }
                1 => {
                    if c.is_numeric(){
                        float_val.push(c);
                        sigfigs+=1+zero_counter;
                        zero_counter=0;
                    }
                    else if c=='E'{stage=2;}
                    else if c=='~'{stage=3;}
                    else {end=i-1; break}
                }
                2 => {
                    if c.is_numeric()||c=='-'{
                        exp.push(c);
                    }
                    else if c=='~'{stage=3;}
                    else {end=i-1; break}
                }
                3 => {
                    if c.is_numeric(){
                        setfigs.push(c);
                    }
                    else {end=i-1; break}
                }
                _ => break
            }
            end=i;
        }
        sigfigs = setfigs.parse::<u32>().unwrap_or(sigfigs);
        let parse_float = float_val.parse::<f32>();
        if parse_float.is_err() {return (end,None)}
        (end,Option::Some(Value{val: parse_float.unwrap()*10_f32.powi(exp.parse::<i8>().unwrap_or(0) as i32),sigfigs}))
    }
}

fn mvf(s: &str) -> MeasuredValue{
    MeasuredValue::from(s)
}

/*
UNITS ^^^^^^^^^^^^^^^



MEASURED VALUES vvvvvvvvvvvvvvvvvvvvvvv
*/

#[derive(Debug, Clone)]
struct MeasuredValue{ 
    value: f32,
    unit: MulUnits,
    sigfigs: u32
}
impl Tokenizable for MeasuredValue{
    type Res = MeasuredValue;
    fn extract(idx: usize, raw: &Vec<char>) -> (usize,Option<MeasuredValue>){
        let ext_val = Value::extract(idx,raw);
        if ext_val.1.is_none() {return (ext_val.0,None)}
        println!("val to unit idx {}",ext_val.0);
        let ext_units = MulUnits::extract(ext_val.0+1,raw);
        let val: Value = ext_val.1.unwrap();
        let res: MeasuredValue = MeasuredValue{value: val.val, unit: ext_units.1.unwrap(), sigfigs: val.sigfigs};
        (ext_units.0,Option::Some(res))
    }
}
impl MeasuredValue{
    fn from(s: &str) -> MeasuredValue{
        if s.len() < 1 {return MeasuredValue::build(0.0,0,MulUnits::default())};
        let mut temp:&str = s.trim_start_matches(|c: char| !c.is_numeric()&&c!='-');
        temp = temp.trim_end();
        let mut float_val: String = String::new();
        let mut end: usize = 0;
        let mut sigfigs: u32 = 0;
        let mut zerocounter: u32 = 0;
        /*
        let mut decimal: bool = false;
        let mut scinote: bool = false;
        let mut setfig: bool = false;*/
        let mut parser: u8 = 0; //0 = normal number, 1 = past decimal place, 2 = sci notation, 3 = set figs
        let mut exp: String = String::new();
        let mut setfig: String = String::new();
        for c in temp.chars(){
            if parser==2 && (c.is_numeric() || c=='-'){
                exp.push(c);
            }else if parser==3 && c.is_numeric(){
                setfig.push(c);
            }else if c.is_numeric() || c == '.' || c=='-'{
                if c.is_numeric(){
                    if c!='0' || (parser==1&&sigfigs!=0) {sigfigs+=1+zerocounter; zerocounter=0}
                    else if sigfigs!=0{
                        zerocounter+=1;
                    }
                }else if c=='.'{
                  sigfigs+=zerocounter;
                  zerocounter=0;
                  parser=1;
                }
                float_val.push(c);
            }else if c=='E'{
                parser=2;
            }else if c=='~'{
                parser=3;
            }else{
                break
            }
            end+=1;  
        }
        if setfig.len()>0 {sigfigs=setfig.parse::<u32>().unwrap_or(sigfigs)};
        let parsedfloat: f32 = float_val.parse::<f32>().unwrap_or(1.0)*10_f32.powf(exp.parse::<f32>().unwrap_or(0.0));
        if parsedfloat==0.0{sigfigs=float_val.len()as u32-1+match float_val.len() {1 => 1, _ => 0}}//consider fixing maybe idk if -0.0 then errors
        MeasuredValue::make(parsedfloat, sigfigs, &s[end..])
    }

    fn make(v: f32, f: u32, s: &str) -> MeasuredValue{
        if s.is_empty(){
            return MeasuredValue{value:v,sigfigs:f,unit: MulUnits::default()}
        }
        let units: MulUnits = s.parse::<MulUnits>().unwrap_or(MulUnits::default());
        MeasuredValue{value: v, unit: units, sigfigs: f}
    }

    fn build(v: f32, f: u32, u: MulUnits) -> MeasuredValue{
        if u.len()==0 {return MeasuredValue{value: v, unit: MulUnits::default(), sigfigs: f}}
        /*
        let mut units: Vec<(String, i8)> = Vec::new();
        let mut repeated: bool = false;
        let len: usize = u.len();
        for n in 0..len{
            let transfer: (String, i8) = u.pop().unwrap();
            for m in 0..units.len(){
                if units[m].0==transfer.0{
                    repeated = true;
                    if units[m].1+transfer.1 != 0 {units[m]=(units[m].0.clone(),units[m].1+transfer.1);}
                    else {units.swap_remove(m);}
                    break;
                }
            }
            if !repeated && transfer.1 != 0 {units.push(transfer);}
            repeated = false;
        }*/
        MeasuredValue{value: v, unit: u, sigfigs: f}
    }

    fn new() -> MeasuredValue{
        MeasuredValue{value: 0.0, unit: MulUnits::default(), sigfigs: u32::MAX-128}
    }
    fn one() -> MeasuredValue{
        MeasuredValue{value: 1.0, unit: MulUnits::default(), sigfigs: u32::MAX-128}
    }
    fn new_const(v: f32) -> MeasuredValue{
        MeasuredValue{value: v, unit: MulUnits::default(), sigfigs:u32::MAX-128}
    }
    //to string with sig figs applied
    fn show(&self) -> String{
        let mut value: String = String::new();
        let unit_str: &str = &self.unit.to_string();
        if self.sigfigs>16{
            value = self.value.to_string();
        }else{
            //let val: f32 = keep_digits(self.value,self.sigfigs+1);
            let val: f32 = round_digits(self.value,self.sigfigs);
            let starting_digit: i32 = starting_pow(self.value);
            //fix
            let least_significant: f32 = ((val%(10_f32.powf((starting_digit-(self.sigfigs as i32)+2) as f32)))*10_f32.powf((self.sigfigs as i32 -2 )as f32)).round();
            if least_significant.is_nan() || least_significant==0.0 || starting_digit.abs()>6{
                let val_shifted: f32 = shift_digits(val,-starting_digit);
                let t: String = val_shifted.to_string();
                let nums: &[u8] = t.as_bytes();
                let mut figs: u32 = 0;
                let mut idx: usize = 0;
                let mut dec: bool = false;
                while figs<self.sigfigs{
                    let mut c: char = '0';
                    if idx<nums.len() {c = nums[idx] as char}
                    else if !dec&&figs==1{c='.'; dec=true};
                    value.push(c);
                    if c.is_numeric() {figs+=1}
                    else if c=='.' {dec=true};
                    idx+=1;
                }
                if starting_digit!=0{
                value.push('E');
                value.push_str(&starting_digit.to_string());}
            }else{
                value = val.to_string();
            }
        }
        format!("{} {}",value,unit_str)
    }
}

impl std::default::Default for MeasuredValue{
    fn default() -> MeasuredValue{
        MeasuredValue{value: 0.0, unit: MulUnits::default(), sigfigs: u32::MAX-128}
    }
}

impl std::str::FromStr for MeasuredValue{
    type Err = MeasuredValue;

    fn from_str(s: &str) -> Result<Self, Self::Err>{
        let res = MeasuredValue::from(s);
        match res.sigfigs{
          0 => Err(MeasuredValue::default()),
          _ => Ok(res)
        }
    }
}
impl ToString for MeasuredValue{
    fn to_string(&self) -> String {
        let unit_str: &str = &self.unit.to_string();
        format!("{}~{} {}",&self.value.to_string(),&self.sigfigs.to_string(),unit_str)
    }
}
impl MeasuredValue{
    fn eval(expr: &str) -> MeasuredValue{
        let fixed: String = expr.to_owned()+"+0";
        let mut val_stack: Vec<MeasuredValue> = Vec::new();
        // first 4 bits = priority, last 4 bits = operator
        // n & 240 to get priority, n & 15 for operator
        let mut op_stack: Vec<u16> = Vec::new();
        
        let mut builder: String = String::new();
        let mut paren: u8 = 0;
        let mut prevchar: char = ' ';

        for c in fixed.chars(){
            if c.is_whitespace(){
                continue
            }
            if c=='(' && val_stack.len() > 0 && make_op(prevchar,0)==0{
                prevchar='*';
            }
            if make_op(prevchar,0)>0{
                if builder.len()==0 || c.is_alphabetic(){
                    if c=='(' && prevchar=='-'{
                        val_stack.push(MeasuredValue::new_const(-1.0));
                        prevchar = '*';
                    }else{
                        builder.push(prevchar);
                        builder.push(c);
                        prevchar = c;
                        continue
                    }
                }else{
                    val_stack.push(builder.parse::<MeasuredValue>().unwrap_or(MeasuredValue::build(-1.0,1,MulUnits::default_error())));
                }
                builder = String::new();
                let op: u16 = make_op(prevchar,paren);
                while op_stack.len()>0 && op_stack.last().unwrap().clone() & 65520 >= op & 65520{
                    let o: u16 = op_stack.pop().unwrap();
                    let right: MeasuredValue = val_stack.pop().unwrap();
                    let left: MeasuredValue = val_stack.pop().unwrap();
                    let val: MeasuredValue = match o & 15 {
                    1 => left+right,
                    2 => left-right,
                    3 => left*right,
                    4 => left/right,
                    _ => left+right,
                    };
                    val_stack.push(val);
                }
                op_stack.push(op);
                if c=='(' {paren+=1} else if c!='-' {builder.push(c)};
            }else if c=='('{
                paren+=1;
            }else if c==')'{
                paren-=1;
            }else{
                if make_op(c,0)==0{
                    builder.push(c);
                }
            }
            prevchar = c;
        }
        val_stack.pop().unwrap()
    }

    fn equat(eq: &str) -> MeasuredValue{
        let fixed: String = eq.replace("=","-(").to_owned()+")+0";
        let mut val_stack: Vec<FormulaVar<MeasuredValue>> = Vec::new();
        // first 4 bits = priority, last 4 bits = operator
        // n & 240 to get priority, n & 15 for operator
        let mut op_stack: Vec<u16> = Vec::new();
        
        let mut builder: String = String::new();
        let mut paren: u8 = 0;
        let mut prevchar: char = ' ';
        for c in fixed.chars(){
            if c.is_whitespace(){
                continue
            }
            if c=='(' && val_stack.len() > 0 && make_op(prevchar,0)==0{
                prevchar='*';
            }
            if make_op(prevchar,0)>0{
                if builder.len()==0 || c.is_alphabetic(){
                    if c=='(' && prevchar == '-'{
                        val_stack.push("-1.0000000000000000000".parse::<FormulaVar<MeasuredValue>>().unwrap());
                        prevchar = '*';
                    }else{
                        builder.push(prevchar);
                        builder.push(c);
                        prevchar = c;
                        continue
                    }
                }else{
                    val_stack.push(builder.parse::<FormulaVar<MeasuredValue>>().unwrap_or(FormulaVar{varcoef: MeasuredValue{value:1.0,unit: MulUnits::default(),sigfigs:u32::MAX-128},val: MeasuredValue::default(),inv:1}));
                }
                builder = String::new();
                let op: u16 = make_op(prevchar,paren);
                while op_stack.len()>0 && op_stack.last().unwrap().clone() & 65520 >= op & 65520{
                    let o: u16 = op_stack.pop().unwrap();
                    let right: FormulaVar<MeasuredValue> = val_stack.pop().unwrap();
                    let left: FormulaVar<MeasuredValue> = val_stack.pop().unwrap();
                    println!("?:{:?} {:?} op:{:?} ?:{} {}",left.varcoef.to_string(),left.val.to_string(),o&15,right.varcoef.to_string(),right.val.to_string());
                    let val: FormulaVar<MeasuredValue> = match o & 15 {
                    1 => left+right,
                    2 => left-right,
                    3 => left*right,
                    4 => left/right,
                    _ => left+right,
                    };
                    println!(" == ?:{:?} {}",val.varcoef.to_string(),val.val.to_string());
                    val_stack.push(val);
                }
                op_stack.push(op);
                if c=='(' {paren+=1} else if c!='-' {builder.push(c)};
            }else if c=='('{
                paren+=1;
            }else if c==')'{
                paren-=1;
            }else{
                if make_op(c,0)==0{
                    builder.push(c);
                }
            }
            prevchar = c;
        }
        val_stack.pop().unwrap().solve()
    }

    fn convert(&mut self, other: &MulUnits){
        self.value = self.unit.convert_val(self.value,other);
    }
}

impl PartialEq for MeasuredValue{
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value && self.unit == other.unit
    }
}

impl ops::Add<MeasuredValue> for MeasuredValue {
    type Output = MeasuredValue;
    fn add(mut self, _rhs: MeasuredValue) -> MeasuredValue {
        let figs: u32 = self.sigfigs.max(_rhs.sigfigs);
        if self==MeasuredValue::default(){
            return _rhs
        }
        if _rhs==MeasuredValue::default(){
            return self
        }
        if self.unit==_rhs.unit {
            return MeasuredValue{value: self.value+_rhs.value, unit: self.unit.clone(), sigfigs: figs}
        }
        if self.unit.same_base_types(&_rhs.unit){
            self.convert(&_rhs.unit);
          return MeasuredValue::build(self.value+_rhs.value,figs,self.unit.clone())
        }
        return MeasuredValue::new();
    }
}
impl ops::Sub<MeasuredValue> for MeasuredValue {
    type Output = MeasuredValue;
    fn sub(self, _rhs: MeasuredValue) -> MeasuredValue {
        let figs: u32 = self.sigfigs.max(_rhs.sigfigs);
        if self==MeasuredValue::default(){
            return MeasuredValue{value: -1.0*_rhs.value, unit: _rhs.unit.clone(), sigfigs: figs}
        }
        if _rhs==MeasuredValue::default(){
            return self
        }
        if self.unit==_rhs.unit {
            return MeasuredValue{value: self.value-_rhs.value, unit: self.unit.clone(), sigfigs: figs}
        }
        return MeasuredValue::new();
    }
}

impl ops::Mul<MeasuredValue> for MeasuredValue {
    type Output = MeasuredValue;
    fn mul(self, _rhs: MeasuredValue) -> MeasuredValue {
        let figs: u32 = self.sigfigs.min(_rhs.sigfigs);
        if self.value*_rhs.value==0.0 {return MeasuredValue::build(0.0,figs, MulUnits::default())}
        return MeasuredValue::build(self.value*_rhs.value, figs, self.unit*_rhs.unit);
    }
}
impl ops::Mul<f32> for MeasuredValue {
    type Output = MeasuredValue;
    fn mul(self, _rhs: f32) -> MeasuredValue {
        if _rhs==0.0 {return MeasuredValue::build(0.0, self.sigfigs, MulUnits::default())}
        return MeasuredValue::build(self.value*_rhs, self.sigfigs, self.unit.clone());
    }
}
impl ops::Mul<i32> for MeasuredValue {
    type Output = MeasuredValue;
    fn mul(self, _rhs: i32) -> MeasuredValue {
        if _rhs==0 {return MeasuredValue::build(0.0, self.sigfigs, MulUnits::default())}
        return MeasuredValue::build(self.value*_rhs as f32, self.sigfigs, self.unit.clone());
    }
}

impl ops::Div<MeasuredValue> for MeasuredValue {
    type Output = MeasuredValue;
    fn div(self, _rhs: MeasuredValue) -> MeasuredValue {
        let figs: u32 = self.sigfigs.min(_rhs.sigfigs);
        if self.value*_rhs.value==0.0 {return MeasuredValue::build(0.0, figs, MulUnits::default())}
        return MeasuredValue::build(self.value/_rhs.value,figs, self.unit/_rhs.unit);
    }
}
impl ops::Div<f32> for MeasuredValue {
    type Output = MeasuredValue;
    fn div(self, _rhs: f32) -> MeasuredValue {
        if _rhs==0.0 {return MeasuredValue::build(self.value*_rhs, self.sigfigs, MulUnits::default())}
        return MeasuredValue::build(self.value/_rhs, self.sigfigs, self.unit.clone());
    }
}
impl ops::Div<i32> for MeasuredValue {
    type Output = MeasuredValue;
    fn div(self, _rhs: i32) -> MeasuredValue {
        if _rhs==0 {return MeasuredValue::build(self.value/_rhs as f32, self.sigfigs, MulUnits::default())}
        return MeasuredValue::build(self.value/_rhs as f32, self.sigfigs, self.unit.clone());
    }
}


enum GasVariables{
    Pressure(Pressure),
    Volume(Volume),
    NumParticles(f32),
    Temp(Temp),
    Constant(f32)
}
impl GasVariables{
    fn from(v: f32, t: &str, u: &str) -> GasVariables{
        match t{
            "p" => GasVariables::Pressure(Pressure::from(v,u)),
            "v" => GasVariables::Volume(Volume::from(v,u)),
            "n" => GasVariables::NumParticles(v),
            "t" => GasVariables::Temp(Temp::from(v,u)),
            _ => GasVariables::Constant(v)
        }
    }
    fn get_num(&self) -> u32{
        match &self {
            GasVariables::Pressure(_) => 0,
            GasVariables::Volume(_) => 1,
            GasVariables::NumParticles(_) => 2,
            GasVariables::Temp(_) => 3,
            GasVariables::Constant(_) => 4
        }
    }
}

enum Temp{
    Kelvin(f32),
    Fahrenheit(f32),
    Celsius(f32)
}
impl Temp{
    fn from(v: f32, t: &str) -> Temp{
        match t{
            "K" => Temp::Kelvin(v),
            "Kelvin" => Temp::Kelvin(v),
            "C" => Temp::Celsius(v),
            "degC" => Temp::Celsius(v),
            "degF" => Temp::Fahrenheit(v),
            "F" => Temp::Fahrenheit(v),
            _ => Temp::Kelvin(v)
        }
    }
    fn to_other(&self, other: Temp) -> Temp{
        match other{
            Temp::Kelvin(_) => self.to_kelvin(),
            Temp::Fahrenheit(_) => self.to_kelvin(), //placeholder
            Temp::Celsius(_) => self.to_celsius(),
        }
    }
    fn to_kelvin(&self) -> Temp{
        match &self{
            Temp::Celsius(v) => Temp::Kelvin(v.clone()-273.15),
            Temp::Fahrenheit(v) => Temp::Kelvin((v.clone()-32.0)*5.0/9.0+273.15),
            Temp::Kelvin(v) => Temp::Kelvin(v.clone())
        }
    }
    fn to_celsius(&self) -> Temp{
        match &self{
            Temp::Celsius(v) => Temp::Celsius(v.clone()),
            Temp::Fahrenheit(v) => Temp::Celsius((v.clone()-32.0)*5.0/9.0),
            Temp::Kelvin(v) => Temp::Celsius(v.clone()+273.15)
        }
    }
    fn get_val(&self) -> f32{
        match &self{
            Temp::Celsius(v) => v+0.0,
            Temp::Fahrenheit(v) => v+0.0,
            Temp::Kelvin(v) => v+0.0
        }
    }
}

enum Pressure{
    atm(f32),
    kPa(f32),
    mmHg(f32),
    Torr(f32),
    psi(f32),
    Pa(f32)
}

impl Pressure{
    fn from(v: f32, t: &str) -> Pressure{
        match t{
            "atm" => Pressure::atm(v),
            "kPa" => Pressure::kPa(v),
            "mmHg" => Pressure::mmHg(v),
            "Torr" => Pressure::Torr(v),
            "psi" => Pressure::psi(v),
            _ => Pressure::atm(v)
        }
    }
    fn to_base_unit(&self) -> f32{
        match self{
            Pressure::kPa(n)=> n*101.325,
            Pressure::mmHg(n) => n*760.0,
            Pressure::Torr(n) => n*760.0,
            Pressure::psi(n) => n*14.695,
            Pressure::Pa(n) => n*101325.0,
            Pressure::atm(n) => n+0.0
        }
    }
    fn to_other(&self, other: &Pressure) -> Pressure{
        match other{
            Pressure::kPa(_)=> Pressure::kPa(self.to_base_unit()/101.325),
            Pressure::mmHg(_) => Pressure::mmHg(self.to_base_unit()/760.0),
            Pressure::Torr(_) => Pressure::Torr(self.to_base_unit()/760.0),
            Pressure::psi(_) => Pressure::psi(self.to_base_unit()/14.695),
            Pressure::Pa(_) => Pressure::Pa(self.to_base_unit()/101325.0),
            Pressure::atm(_) => Pressure::Pa(self.to_base_unit()/1.0)
        }
    }
    fn get_val(&self) -> f32{
        match self{
            Pressure::kPa(n)=> n+0.0,
            Pressure::mmHg(n) => n+0.0,
            Pressure::Torr(n) => n+0.0,
            Pressure::psi(n) => n+0.0,
            Pressure::Pa(n) => n+0.0,
            Pressure::atm(n) => n+0.0
        }
    }
}

enum Volume{
    mL(f32),
    L(f32),
    cm3(f32),
}
impl Volume{
    fn from(v: f32, t: &str) -> Volume{
        match t{
            "mL" => Volume::mL(v),
            "L" => Volume::L(v),
            "cm3" => Volume::cm3(v),
            _ => Volume::L(v)
        }
    }

    fn to_base_unit(&self) -> f32 {
        match self{
            Volume::mL(n) => n*1000.0,
            Volume::L(n) => n*1.0,
            Volume::cm3(n) => n*1000.0
        }
    }
    fn to_other(&self, other: &Volume) -> Volume{
        match other{
            Volume::mL(_) => Volume::mL(self.to_base_unit()/1000.0),
            Volume::L(_) => Volume::L(1.0*self.to_base_unit()),
            Volume::cm3(_) => Volume::cm3(1000.0*self.to_base_unit()/1000.0)
        }
    }
    fn get_val(&self) -> f32 {
        match self{
            Volume::mL(n) => n+0.0,
            Volume::L(n) => n+0.0,
            Volume::cm3(n) => n+0.0
        }
    }
}
/*
fn parse_gas(input: &str) -> String{
    let mut pressure: (Option<Pressure>,Option<Pressure>) = (None,None);
    let mut volume: (Option<Volume>, Option<Volume>) = (None,None);
    let mut temp: (Option<Temp>, Option<Temp>) = (None,None);
    let mut num: (Option<f32>, Option<f32>) = (None,None);
    let mut r: Option<f32>;
    let mut customConst: bool = false;
    let mut varcount: (u32,u32) = (0,0);
    let mut builder: String = String::from("");
    for (p,c) in input.char_indices(){
        if c=='=' {
            match &input[p-1..p]{
                "1" => {customConst = true; varcount.0+=1; if builder.len()>2 {builder=builder[0..builder.len()-2].to_string(); let t = parse_gas_tuple(&builder); match t{GasVariables::Pressure(p) => pressure.0=Some(p), GasVariables::Volume(v) => volume.0=Some(v), GasVariables::Temp(te)=>temp.0=Some(te), GasVariables::NumParticles(n)=>num.0=Some(n), GasVariables::Constant(c) => r=Some(c)};builder = input[p-2..p+1].to_string()}}
                "2" => {customConst = true; varcount.1+=1; if builder.len()>2 {builder=builder[0..builder.len()-2].to_string(); let t = parse_gas_tuple(&builder); match t{GasVariables::Pressure(p) => pressure.1=Some(p), GasVariables::Volume(v) => volume.1=Some(v), GasVariables::Temp(te)=>temp.1=Some(te), GasVariables::NumParticles(n)=>num.1=Some(n), GasVariables::Constant(c) => r=Some(c)}; builder = input[p-2..p+1].to_string()}}
                _ => {if builder.len()>1 {builder=builder[0..builder.len()-1].to_string(); let t = parse_gas_tuple(&builder); match t{GasVariables::Pressure(p) => pressure.0=Some(p), GasVariables::Volume(v) => volume.0=Some(v), GasVariables::Temp(te)=>temp.0=Some(te), GasVariables::NumParticles(n)=>num.0=Some(n), GasVariables::Constant(c) => r=Some(c)}; builder = input[p-1..p+1].to_string()}}
            }
        }else{
            builder += &c.to_string();
        }
    }
    if customConst {
        //p,v,n,t,r
        let mut data: [f32; 5] = [-1.0,-1.0,-1.0,-1.0,-1.0];
        let mut other: [f32; 4] = [-1.0,-1.0,-1.0,-1.0];
        let mut unknown: GasConstants;
        let mut pressureUnit: Option<Pressure>;
        let mut tempUnit: Option<Temp>;
        let mut volumeUnit: Option<Volume>;
        let mut rval: f32 = 1.0;
        let mut notgiven: u8 = 0;
        if varcount.0<varcount.1 {
            match pressure.1{
                Some(mut p)=>{rval*=p.get_val(); pressureUnit=Some(p)},
                None=>notgiven+=8
            }
            match volume.1{
                Some(mut v)=>{rval*=v.get_val(); volumeUnit=Some(v)  },
                None=>notgiven+=4
            }
            match num.1{
                Some(mut n)=>{rval*=1.0/n},
                None=>notgiven+=2
            }
            match temp.1{
                Some(mut t)=>{rval*=1.0/t.get_val(); tempUnit=Some(t)},
                None=>notgiven+=1
            }
        }else{

        }
    }
    return "ea".to_string()
}
fn parse_gas_tuple(input: &str) -> GasVariables{
    let mut floatval: String = String::new();
        let mut unit: String = String::new();
        for (p,c) in input[2..].char_indices(){
            if c.is_numeric()||c=='.' {floatval+=&c.to_string()}
            else if c.is_alphabetic() {unit+=&c.to_string()}
        }
    GasVariables::from(floatval.parse().unwrap(),&input.trim()[0..1],&unit)
}*/

enum Msg{
    AddOne,
    ShowText(String),
    UpdateFields(usize,String),
    None,
}

struct CounterComponent{
    count: i32,
    text: String,
    input_fields: Vec<String>,
    prev_answers: Vec<String>, //previous unformatted answers
    prev_results: Vec<String>, //previous formatted results
    prev_solution: Vec<String>, //previous solution with work (if applicable, otherwise length = 1)
    store: Storage
}

impl Component for CounterComponent{
    type Message = Msg;
    type Properties = ();
    fn create(_ctx: &Context<Self>) -> Self{
    Self{count: 0, prev_answers: Vec::new(), input_fields: Vec::new(), prev_results: Vec::new(),prev_solution: Vec::new(), text: String::new(), store: Storage::make()}
    }
    fn update(&mut self, _ctx: &Context<Self>, msg: Self::Message) -> bool{
        match msg{
            Msg::AddOne =>{
                self.count+=1;
                if !self.prev_solution.is_empty() {self.prev_results.push(self.prev_solution.pop().unwrap_or(String::default()))};
                self.prev_solution = Vec::new();
                self.text=self.text.trim().to_string();
                //self.answer=self.text.clone()+" = ";
                if self.text.is_empty(){
                    return false
                }else if self.text.starts_with("idealgas"){
                    //self.answer = parse_gas(&self.text);
                }else if self.text.contains("=") {
                    let res=MeasuredValue::equat(&self.text);
                    self.prev_solution.push(self.text.clone()+", ? = "+&res.to_string()+" ~> "+&res.show());
                }else{
                    if self.text.starts_with(|c: char| c.is_numeric()||c=='-'){
                        let res=MeasuredValue::eval(&self.text);
                        self.prev_solution.push(self.text.clone()+" = "+&res.to_string() + " ~> "+&res.show());
                    }else{
                        return false
                    }
                }
                true
            }
            Msg::ShowText(content) =>{
                self.text = content;
                //self.hello_world = pressureConverter(5.0,"kPa").to_string();
                false
            }
            Msg::UpdateFields(idx,content) =>{
                self.input_fields[idx]=content;
                false
            }
            Msg::None =>{
                //PreviousInput::create(yew::);
                false
            }
        }
    }
    fn view(&self, _ctx: &Context<Self>) -> Html{
        let link = _ctx.link();
        let p_v: &Vec<String> = &self.prev_results;
        let p_s: &Vec<String> = &self.prev_solution;
        html! {
            <>
                <div class = "i">
                    <h1> {"dr v solver"} <br/> </h1>
                    <p> {"only the 'eval' solver works right now. it is literally just a glorified calculator with units and sig figs (but addition/subtraction sig figs doesn't even work)"}
                    <br/> {"supports sig figs, basic operations with values and units, solving for unknowns (denoted by a ?) in an equation"}
                    <br/> {"unit conversions sometimes works but only with addition. identify units with _, for example 1.0 mol_H2O, but this doesn't actually do anything other than crash occasionally"}
                    <br/> {"note that parenthesis are not supported when writing units, and everything after a '/' is assumed to be in parenthesis, so J/(g*C) should be written as J/g*C"}
                    <br/> {"example: 5.0g*4.184J/g*C*(20.0C-10.0C) will solve, and 10.0=5.0+? will give 5.0 for ? as an answer"}
                    <br/> {"also if you can't be bothered to convert g to mol by hand then just do 0.000mol_H2O + ###g_H2O or mol -> g with 0.000g_H2O + ###mol_H2O"}
                    <br/> {"anyways here are some constants to copy paste if you need them and also im too lazy to actually code:"}
                    <br/> {"0.08206~20 L*atm/mol*K, 4.184J/g*C"}
                    <br/> <br/> </p>
                </div>
                <div class = "dont">
                    if p_v.len()>0||p_s.len()>0{
                    {
                        p_v.iter().enumerate().map(|(i, v)| {
                            html!{<div key={i}>{format!("{}) {}",i,v.clone())}</div>}
                        }).collect::<Html>()
                    }
                    <br/><pre class="know">
                    <p> {"Solution:"} </p>
                    {
                        p_s.iter().enumerate().map(|(i, v)| {
                            html!{<div key={i}>{format!("{}) {}",i+1,v.clone())}</div>}
                        }).collect::<Html>()
                    }
                    </pre>
                    <br/>
                    }
                    //<p> <br/> {self.text.clone()} </p>
                </div>
                <div class = "html">
                    <br/>
                    <label for="fname">{"type into the box: "}</label>
                    <input type="text" id="fname" name="fname" oninput={link.callback(|event: InputEvent| {let input: HtmlInputElement = event.target_unchecked_into(); Msg::ShowText(input.value())})} onkeypress={link.callback(|key:KeyboardEvent| {if key.char_code()==13 {Msg::AddOne} else{Msg::None}})}/>
                    <button onclick={link.callback(|_| Msg::AddOne)}> {"solve"}</button>
                    <p> {self.count} </p>
                </div>
            </>
        }
    }
}
struct PreviousInput{
    input: String,
    answer: String
}


struct GasConstants{

}

//stores constant variables, such as periodic table, or polyatomics
struct Storage{
    table: PeriodicTable,
    polyatomics: Vec<Polyatomic>,
    ions: Vec<Option<i8>>,
}

impl Storage{
    fn make() -> Storage{
        Storage { 
            table: PeriodicTable::build(), 
            polyatomics: Vec::new(),
            ions: Vec::new()
        }
    }
    fn make_polyatomic(){

    }
    
}

struct Element{
    atomic_number: u8,
    molar_mass: f32,
    symbol: String,
    name: String,
}

impl Element{
    fn make(atomic_number: u8, molar_mass: f32, symbol: &str, name: &str) -> Element{
        Element{
            atomic_number,
            molar_mass,
            symbol: symbol.to_string(),
            name: name.to_string()
        }
    }
    fn to_moles(&self, mass: MeasuredValue)-> MeasuredValue{
        if mass.unit.len()==1 && mass.unit.same_varients(vec![(UnitType::Mass,1)]){ //make accept all masses
            return MeasuredValue::build(mass.value/self.molar_mass.clone(),mass.sigfigs,MulUnits{units: vec![Unit::from_str(&format!("mol_{}",self.symbol)).unwrap()]})
        }
        MeasuredValue{value:-1.0, unit: MulUnits{units: vec![Unit::from_str("not mass").unwrap()]},sigfigs: 0}
    }
    fn to_grams(&self, moles: MeasuredValue)-> MeasuredValue{
        if moles.unit.len()==1 && moles.unit.units[0].unit=="mol"{
            return moles*self.molar_mass.clone()
        }
        MeasuredValue{value:-1.0, unit: MulUnits{units: vec![Unit::from_str("not mol").unwrap()]},sigfigs:0}
    }
}

struct Polyatomic{
    //element atomic number, num of that element
    molar_mass: f32,
    name: String,
    elements: Vec<(u8,u8)>
}

impl Polyatomic{
    //hashes are generated by the base 128 representation of 
    fn get_hash(&self) -> u32{
        329
    }
}

struct IonicCompound{
    
} 

struct MolecularCompound{


}


struct PeriodicTable{
    elements: [Element; 109]
}

impl PeriodicTable{
    fn get_elem(&self, atomic_number : u8) -> &Element{
        &self.elements[atomic_number as usize]
    }

    fn build() -> PeriodicTable{
        PeriodicTable { 
            elements: [
                Element::make(1,1.01,"H","Hydrogen"),
                Element::make(2,4.00,"He","Helium"),
                Element::make(3,6.94,"Li","Lithium"),
                Element::make(4,9.01,"Be","Beryllium"),
                Element::make(5,10.81,"B","Boron"),
                Element::make(6,12.01,"C","Carbon"),
                Element::make(7,14.01,"N","Nitrogen"),
                Element::make(8,16.00,"O","Oxygen"),
                Element::make(9,19.00,"F","Fluorine"),
                Element::make(10,20.18,"Ne","Neon"),
                Element::make(11,22.99,"Na","Sodium"),
                Element::make(12,24.31,"Mg","Magnesium"),
                Element::make(13,26.98,"Al","Aluminum"),
                Element::make(14,28.09,"Si","Silicon"),
                Element::make(15,30.97,"P","Phosphorus"),
                Element::make(16,32.06,"S","Sulfur"),
                Element::make(17,35.45,"Cl","Chlorine"),
                Element::make(18,39.95,"Ar","Argon"),
                Element::make(19,39.1,"K","Potassium"),
                Element::make(20,40.08,"Ca","Calcium"),
                Element::make(21,44.96,"Sc","Scandium"),
                Element::make(22,47.9,"Ti","Titanium"),
                Element::make(23,50.94,"V","Vanadium"),
                Element::make(24,52.00,"Cr","Chromium"),
                Element::make(25,54.94,"Mn","Manganese"),
                Element::make(26,55.85,"Fe","Iron"),
                Element::make(27,58.93,"Co","Cobalt"),
                Element::make(28,58.7,"Ni","Nickel"),
                Element::make(29,63.55,"Cu","Copper"),
                Element::make(30,65.38,"Zn","Zinc"),
                Element::make(31,69.72,"Ga","Gallium"),
                Element::make(32,72.59,"Ge","Germanium"),
                Element::make(33,74.92,"As","Arsenic"),
                Element::make(34,78.96,"Se","Selenium"),
                Element::make(35,79.9,"Br","Bromine"),
                Element::make(36,83.8,"Kr","Krypton"),
                Element::make(37,85.47,"Rb","Rubidium"),
                Element::make(38,87.62,"Sr","Strontium"),
                Element::make(39,88.91,"Y","Yttrium"),
                Element::make(40,91.22,"Zr","Zirconium"),
                Element::make(41,92.91,"Nb","Niobium"),
                Element::make(42,95.94,"Mo","Molybdenum"),
                Element::make(43,98.00,"Tc","Technetium"),
                Element::make(44,101.07,"Ru","Ruthenium"),
                Element::make(45,102.91,"Rh","Rhodium"),
                Element::make(46,106.4,"Pd","Palladium"),
                Element::make(47,107.87,"Ag","Silver"),
                Element::make(48,112.41,"Cd","Cadmium"),
                Element::make(49,114.82,"In","Indium"),
                Element::make(50,118.69,"Sn","Tin"),
                Element::make(51,121.75,"Sb","Antimony"),
                Element::make(52,127.6,"Te","Tellurium"),
                Element::make(53,126.9,"I","Iodine"),
                Element::make(54,131.3,"Xe","Xenon"),
                Element::make(55,132.91,"Cs","Cesium"),
                Element::make(56,137.33,"Ba","Barium"),
                Element::make(57,138.91,"La","Lanthanum"),
                Element::make(58,140.12,"Ce","Cerium"),
                Element::make(59,140.91,"Pr","Praseodymium"),
                Element::make(60,144.24,"Nd","Neodymium"),
                Element::make(61,145.00,"Pm","Promethium"),
                Element::make(62,150.4,"Sm","Samarium"),
                Element::make(63,151.96,"Eu","Europium"),
                Element::make(64,157.25,"Gd","Gadolinium"),
                Element::make(65,158.93,"Tb","Terbium"),
                Element::make(66,162.5,"Dy","Dysprosium"),
                Element::make(67,164.93,"Ho","Holmium"),
                Element::make(68,167.26,"Er","Erbium"),
                Element::make(69,168.93,"Tm","Thulium"),
                Element::make(70,173.04,"Yb","Ytterbium"),
                Element::make(71,174.97,"Lu","Lutetium"),
                Element::make(72,178.49,"Hf","Hafnium"),
                Element::make(73,180.95,"Ta","Tantalum"),
                Element::make(74,183.85,"W","Tungsten"),
                Element::make(75,186.21,"Re","Rhenium"),
                Element::make(76,190.2,"Os","Osmium"),
                Element::make(77,192.22,"Ir","Iridium"),
                Element::make(78,195.09,"Pt","Platinum"),
                Element::make(79,196.97,"Au","Gold"),
                Element::make(80,200.59,"Hg","Mercury"),
                Element::make(81,204.37,"Tl","Thallium"),
                Element::make(82,207.2,"Pb","Lead"),
                Element::make(83,208.98,"Bi","Bismuth"),
                Element::make(84,209.0,"Po","Polonium"),
                Element::make(85,210.0,"At","Astatine"),
                Element::make(86,222.0,"Rn","Radon"),
                Element::make(87,223.0,"Fr","Francium"),
                Element::make(88,226.03,"Ra","Radium"),
                Element::make(89,227.03,"Ac","Actinium"),
                Element::make(90,232.04,"Th","Thorium"),
                Element::make(91,231.04,"Pa","Protactinium"),
                Element::make(92,238.03,"U","Uranium"),
                Element::make(93,237.05,"Np","Neptunium"),
                Element::make(94,242.0,"Pu","Plutonium"),
                Element::make(95,243.0,"Am","Americium"),
                Element::make(96,247.0,"Cm","Curium"),
                Element::make(97,247.0,"Bk","Berkelium"),
                Element::make(98,251.0,"Cf","Californium"),
                Element::make(99,252.0,"Es","Einsteinium"),
                Element::make(100,257.0,"Fm","Fermium"),
                Element::make(101,258.0,"Md","Mendelevium"),
                Element::make(102,250.0,"No","Nobelium"),
                Element::make(103,260.0,"Lr","Lawrencium"),
                Element::make(104,261.0,"Rf","Rutherfordium"),
                Element::make(105,262.0,"Db","Dubnium"),
                Element::make(106,263.0,"Sg","Seaborgium"),
                Element::make(107,262.0,"Bh","Bohrium"),
                Element::make(108,255.0,"Hs","Hassium"),
                Element::make(109,256.0,"Mt","Meitnerium"),
            ]
        }
    }

    fn elem_from_symbol(&self, symbol: &str) -> &Element{
        &self.elements[match symbol{
            "H" => 0,
            "He" => 1,
            "Li" => 3,
            "Be" => 4,
            "B" => 5,
            "C" => 6,
            "N" => 7,
            "O" => 8,
            "F" => 9,
            "Ne" => 10,
            "Na" => 11,
            "Mg" => 12,
            "Al" => 13,
            "Si" => 14,
            "P" => 15,
            "S" => 16,
            "Cl" => 17,
            "Ar" => 18,
            "K" => 19,
            "Ca" => 20,
            "Sc" => 21,
            "Ti" => 22,
            "V" => 23,
            "Cr" => 24,
            "Mn" => 25,
            "Fe" => 26,
            "Co" => 27,
            "Ni" => 28,
            "Cu" => 29,
            "Zn" => 30,
            "Ga" => 31,
            "Ge" => 32,
            "As" => 33,
            "Se" => 34,
            "Br" => 35,
            "Kr" => 36,
            "Rb" => 37,
            "Sr" => 38,
            "Y" => 39,
            "Zr" => 40,
            "Nb" => 41,
            "Mo" => 42,
            "Tc" => 43,
            "Ru" => 44,
            "Rh" => 45,
            "Pd" => 46,
            "Ag" => 47,
            "Cd" => 48,
            "In" => 49,
            "Sn" => 50,
            "Sb" => 51,
            "Te" => 52,
            "I" => 53,
            "Xe" => 54,
            "Cs" => 55,
            "Ba" => 56,
            "La" => 57,
            "Ce" => 58,
            "Pr" => 59,
            "Nd" => 60,
            "Pm" => 61,
            "Sm" => 62,
            "Eu" => 63,
            "Gd" => 64,
            "Tb" => 65,
            "Dy" => 66,
            "Ho" => 67,
            "Er" => 68,
            "Tm" => 69,
            "Yb" => 70,
            "Lu" => 71,
            "Hf" => 72,
            "Ta" => 73,
            "W" => 74,
            "Re" => 75,
            "Os" => 76,
            "Ir" => 77,
            "Pt" => 78,
            "Au" => 79,
            "Hg" => 80,
            "Tl" => 81,
            "Pb" => 82,
            "Bi" => 83,
            "Po" => 84,
            "At" => 85,
            "Rn" => 86,
            "Fr" => 87,
            "Ra" => 88,
            "Ac" => 89,
            "Th" => 90,
            "Pa" => 91,
            "U" => 92,
            "Np" => 93,
            "Pu" => 94,
            "Am" => 95,
            "Cm" => 96,
            "Bk" => 97,
            "Cf" => 98,
            "Es" => 99,
            "Fm" => 100,
            "Md" => 101,
            "No" => 102,
            "Lr" => 103,
            "Rf" => 104,
            "Db" => 105,
            "Sg" => 106,
            "Bh" => 107,
            "Hs" => 108,
            "Mt" => 109,
            _ => 0
        }]
    }
}

impl Element{
    fn from(symbol: &str) -> Element{
        multi_matcher!(symbol {
                Element::make(1,1.01,"H","Hydrogen"),("H","Hydrogen"),
                Element::make(2,4.0,"He","Helium"),("He","Helium"),
                Element::make(3,6.94,"Li","Lithium"),("Li","Lithium"),
                Element::make(4,9.01,"Be","Beryllium"),("Be","Beryllium"),
                Element::make(5,10.81,"B","Boron"),("B","Boron"),
                Element::make(6,12.01,"C","Carbon"),("C","Carbon"),
                Element::make(7,14.01,"N","Nitrogen"),("N","Nitrogen"),
                Element::make(8,16.0,"O","Oxygen"),("O","Oxygen"),
                Element::make(9,19.0,"F","Fluorine"),("F","Fluorine"),
                Element::make(10,20.18,"Ne","Neon"),("Ne","Neon"),
                Element::make(11,22.99,"Na","Sodium"),("Na","Sodium"),
                Element::make(12,24.31,"Mg","Magnesium"),("Mg","Magnesium"),
                Element::make(13,26.98,"Al","Aluminum"),("Al","Aluminum"),
                Element::make(14,28.09,"Si","Silicon"),("Si","Silicon"),
                Element::make(15,30.97,"P","Phosphorus"),("P","Phosphorus"),
                Element::make(16,32.06,"S","Sulfur"),("S","Sulfur"),
                Element::make(17,35.45,"Cl","Chlorine"),("Cl","Chlorine"),
                Element::make(18,39.95,"Ar","Argon"),("Ar","Argon"),
                Element::make(19,39.1,"K","Potassium"),("K","Potassium"),
                Element::make(20,40.08,"Ca","Calcium"),("Ca","Calcium"),
                Element::make(21,44.96,"Sc","Scandium"),("Sc","Scandium"),
                Element::make(22,47.9,"Ti","Titanium"),("Ti","Titanium"),
                Element::make(23,50.94,"V","Vanadium"),("V","Vanadium"),
                Element::make(24,52.0,"Cr","Chromium"),("Cr","Chromium"),
                Element::make(25,54.94,"Mn","Manganese"),("Mn","Manganese"),
                Element::make(26,55.85,"Fe","Iron"),("Fe","Iron"),
                Element::make(27,58.93,"Co","Cobalt"),("Co","Cobalt"),
                Element::make(28,58.7,"Ni","Nickel"),("Ni","Nickel"),
                Element::make(29,63.55,"Cu","Copper"),("Cu","Copper"),
                Element::make(30,65.38,"Zn","Zinc"),("Zn","Zinc"),
                Element::make(31,69.72,"Ga","Gallium"),("Ga","Gallium"),
                Element::make(32,72.59,"Ge","Germanium"),("Ge","Germanium"),
                Element::make(33,74.92,"As","Arsenic"),("As","Arsenic"),
                Element::make(34,78.96,"Se","Selenium"),("Se","Selenium"),
                Element::make(35,79.9,"Br","Bromine"),("Br","Bromine"),
                Element::make(36,83.8,"Kr","Krypton"),("Kr","Krypton"),
                Element::make(37,85.47,"Rb","Rubidium"),("Rb","Rubidium"),
                Element::make(38,87.62,"Sr","Strontium"),("Sr","Strontium"),
                Element::make(39,88.91,"Y","Yttrium"),("Y","Yttrium"),
                Element::make(40,91.22,"Zr","Zirconium"),("Zr","Zirconium"),
                Element::make(41,92.91,"Nb","Niobium"),("Nb","Niobium"),
                Element::make(42,95.94,"Mo","Molybdenum"),("Mo","Molybdenum"),
                Element::make(43,98.0,"Tc","Technetium"),("Tc","Technetium"),
                Element::make(44,101.07,"Ru","Ruthenium"),("Ru","Ruthenium"),
                Element::make(45,102.91,"Rh","Rhodium"),("Rh","Rhodium"),
                Element::make(46,106.4,"Pd","Palladium"),("Pd","Palladium"),
                Element::make(47,107.87,"Ag","Silver"),("Ag","Silver"),
                Element::make(48,112.41,"Cd","Cadmium"),("Cd","Cadmium"),
                Element::make(49,114.82,"In","Indium"),("In","Indium"),
                Element::make(50,118.69,"Sn","Tin"),("Sn","Tin"),
                Element::make(51,121.75,"Sb","Antimony"),("Sb","Antimony"),
                Element::make(52,127.6,"Te","Tellurium"),("Te","Tellurium"),
                Element::make(53,126.9,"I","Iodine"),("I","Iodine"),
                Element::make(54,131.3,"Xe","Xenon"),("Xe","Xenon"),
                Element::make(55,132.91,"Cs","Cesium"),("Cs","Cesium"),
                Element::make(56,137.33,"Ba","Barium"),("Ba","Barium"),
                Element::make(57,138.91,"La","Lanthanum"),("La","Lanthanum"),
                Element::make(58,140.12,"Ce","Cerium"),("Ce","Cerium"),
                Element::make(59,140.91,"Pr","Praseodymium"),("Pr","Praseodymium"),
                Element::make(60,144.24,"Nd","Neodymium"),("Nd","Neodymium"),
                Element::make(61,145.0,"Pm","Promethium"),("Pm","Promethium"),
                Element::make(62,150.4,"Sm","Samarium"),("Sm","Samarium"),
                Element::make(63,151.96,"Eu","Europium"),("Eu","Europium"),
                Element::make(64,157.25,"Gd","Gadolinium"),("Gd","Gadolinium"),
                Element::make(65,158.93,"Tb","Terbium"),("Tb","Terbium"),
                Element::make(66,162.5,"Dy","Dysprosium"),("Dy","Dysprosium"),
                Element::make(67,164.93,"Ho","Holmium"),("Ho","Holmium"),
                Element::make(68,167.26,"Er","Erbium"),("Er","Erbium"),
                Element::make(69,168.93,"Tm","Thulium"),("Tm","Thulium"),
                Element::make(70,173.04,"Yb","Ytterbium"),("Yb","Ytterbium"),
                Element::make(71,174.97,"Lu","Lutetium"),("Lu","Lutetium"),
                Element::make(72,178.49,"Hf","Hafnium"),("Hf","Hafnium"),
                Element::make(73,180.95,"Ta","Tantalum"),("Ta","Tantalum"),
                Element::make(74,183.85,"W","Tungsten"),("W","Tungsten"),
                Element::make(75,186.21,"Re","Rhenium"),("Re","Rhenium"),
                Element::make(76,190.2,"Os","Osmium"),("Os","Osmium"),
                Element::make(77,192.22,"Ir","Iridium"),("Ir","Iridium"),
                Element::make(78,195.09,"Pt","Platinum"),("Pt","Platinum"),
                Element::make(79,196.97,"Au","Gold"),("Au","Gold"),
                Element::make(80,200.59,"Hg","Mercury"),("Hg","Mercury"),
                Element::make(81,204.37,"Tl","Thallium"),("Tl","Thallium"),
                Element::make(82,207.2,"Pb","Lead"),("Pb","Lead"),
                Element::make(83,208.98,"Bi","Bismuth"),("Bi","Bismuth"),
                Element::make(84,209.0,"Po","Polonium"),("Po","Polonium"),
                Element::make(85,210.0,"At","Astatine"),("At","Astatine"),
                Element::make(86,222.0,"Rn","Radon"),("Rn","Radon"),
                Element::make(87,223.0,"Fr","Francium"),("Fr","Francium"),
                Element::make(88,226.03,"Ra","Radium"),("Ra","Radium"),
                Element::make(89,227.03,"Ac","Actinium"),("Ac","Actinium"),
                Element::make(90,232.04,"Th","Thorium"),("Th","Thorium"),
                Element::make(91,231.04,"Pa","Protactinium"),("Pa","Protactinium"),
                Element::make(92,238.03,"U","Uranium"),("U","Uranium"),
                Element::make(93,237.05,"Np","Neptunium"),("Np","Neptunium"),
                Element::make(94,242.0,"Pu","Plutonium"),("Pu","Plutonium"),
                Element::make(95,243.0,"Am","Americium"),("Am","Americium"),
                Element::make(96,247.0,"Cm","Curium"),("Cm","Curium"),
                Element::make(97,247.0,"Bk","Berkelium"),("Bk","Berkelium"),
                Element::make(98,251.0,"Cf","Californium"),("Cf","Californium"),
                Element::make(99,252.0,"Es","Einsteinium"),("Es","Einsteinium"),
                Element::make(100,257.0,"Fm","Fermium"),("Fm","Fermium"),
                Element::make(101,258.0,"Md","Mendelevium"),("Md","Mendelevium"),
                Element::make(102,250.0,"No","Nobelium"),("No","Nobelium"),
                Element::make(103,260.0,"Lr","Lawrencium"),("Lr","Lawrencium"),
                Element::make(104,261.0,"Rf","Rutherfordium"),("Rf","Rutherfordium"),
                Element::make(105,262.0,"Db","Dubnium"),("Db","Dubnium"),
                Element::make(106,263.0,"Sg","Seaborgium"),("Sg","Seaborgium"),
                Element::make(107,262.0,"Bh","Bohrium"),("Bh","Bohrium"),
                Element::make(108,255.0,"Hs","Hassium"),("Hs","Hassium"),
                Element::make(109,256.0,"Mt","Meitnerium"),("Mt","Meitnerium"),
                _ Element::make(0,0.0,"","")
            }
        )
    }
    fn molar_mass(substance: &str) -> f32{
        let mut elem: String = String::new();
        let mut quant: String = String::new();
        let mut res: f32 = 0.0;
        let mut substance: String = substance.to_string();
        substance.push('G');
        for c in substance.chars(){
            if c.is_uppercase(){
                res+=Element::from(&elem).molar_mass*quant.parse::<f32>().unwrap_or(1.0);
                elem=String::new();
                quant=String::new();
                elem.push(c);
            }else if c.is_numeric(){
                quant.push(c);
            }else if c.is_lowercase(){
                elem.push(c);
            }
        }
        res
    }
}