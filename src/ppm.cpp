#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>

namespace nb = nanobind;

#define _USE_MATH_DEFINES
#include <math.h>

#include <random>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <functional>
#include <iterator>
#include <string>
#include <unordered_map>
#include <boost/functional/hash.hpp>

typedef std::vector<int> sequence;

sequence subseq(const sequence &x, unsigned int first, unsigned int last) {
  if (last >= x.size() || last < first) {
    throw std::invalid_argument("invalid subsequence indices");
  }
  int n = 1 + last - first;
  sequence res(n);
  for (int j = 0; j < n; j ++) {
    res[j] = x[j + first];
  }
  return(res);
}

sequence last_n (const sequence &x, int n) {
  if (n < 0) {
    throw std::invalid_argument("n cannot be less than 0");
  }
  int original_length = static_cast<int>(x.size());
  if (n > original_length) {
    throw std::invalid_argument("cannot excise more elements than the sequence contains");
  }
  if (n == 0) {
    sequence res(0);
    return res;
  } else {
    sequence res(n);
    for (int i = 0; i < n; i ++) {
      res[i] = x[i + original_length - n];
    }
    return res;
  }
}

void print(const sequence &x) {
  for (unsigned int j = 0; j < x.size(); j ++) {
    if (j > 0) {
      std::cout << " ";
    }
    std::cout << x[j];
  }
  std::cout << "\n";
}

void print(const std::vector<double> &x) {
  for (unsigned int j = 0; j < x.size(); j ++) {
    if (j > 0) {
      std::cout << " ";
    }
    std::cout << x[j];
  }
  std::cout << "\n";
}

void print(const std::vector<bool> &x) {
  for (unsigned int j = 0; j < x.size(); j ++) {
    if (j > 0) {
      std::cout << " ";
    }
    if (x[j]) {
      std::cout << "True";
    } else {
      std::cout << "False";
    }
  }
  std::cout << "\n";
}

template <typename Container> // we can make this generic for any container [1]
struct container_hash {
  std::size_t operator() (Container const& c) const {
    return boost::hash_range(c.begin(), c.end());
  }
};

class record {
  
};

class record_simple: public record {
public: 
  long int full_count = 0;
  long int up_ex_count = 0;
  
  record_simple() {};
  
  void add_1(bool full_only) {
    if (full_count >= LONG_MAX || up_ex_count >= LONG_MAX) {
      throw std::range_error("cannot increment this record count any higher");
    }
    full_count ++;
    if (!full_only) {
      up_ex_count ++;
    }
  }
};

double compute_entropy(std::vector<double> x) {
  int n = x.size();
  double counter = 0;
  for (int i = 0; i < n; i ++) {
    double p = x[i];
    counter -= p * log2(p);
  }
  return counter;
}

std::vector<double> normalise_distribution(std::vector<double> &x) {
  double total = 0;
  int n = static_cast<int>(x.size());
  for (int i = 0; i < n; i ++) {
    total += x[i];
  }
  for (int i = 0; i < n; i ++) {
    x[i] = x[i] / total;
  }
  return x;
}

class record_decay: public record {
public:
  record_decay() {}
  std::vector<int> pos;
  // std::vector<double> time;
  void insert(int pos_, double time_) {
    pos.push_back(pos_);
    // time.push_back(time_);
  }
};

class symbol_prediction {
public:
  int symbol;
  int pos;
  double time;
  int model_order;
  std::vector<double> distribution;
  double information_content;
  
  symbol_prediction(int symbol_, 
                    int pos_, 
                    double time_, 
                    int model_order_,
                    const std::vector<double> &distribution_) {
    int dist_size_ = distribution_.size();
    if (symbol_ > dist_size_) {
      std::cout << "symbol = " << symbol_ << ", distribution(n) = " << distribution_.size() << "\n";
      throw std::range_error("observed symbol not compatible with distribution dimensions");
    }
    
    symbol = symbol_;
    pos = pos_;
    time = time_;
    model_order = model_order_;
    distribution = distribution_;
    information_content = - log2(distribution[symbol]);
  }
};

class sequence_prediction {
public: 
  bool return_distribution;
  bool return_entropy;
  bool decay;
  
  std::vector<int> symbol;
  std::vector<int> pos;
  std::vector<double> time;
  std::vector<int> model_order;
  std::vector<double> information_content;
  std::vector<double> entropy;
  std::vector<std::vector<double>> distribution;
  
  sequence_prediction(bool return_distribution_,
                      bool return_entropy_,
                      bool decay_) {
    return_distribution = return_distribution_;
    return_entropy = return_entropy_;
    decay = decay_;
  }
  
  void insert(const symbol_prediction &x) {
    symbol.push_back(x.symbol);
    model_order.push_back(x.model_order);
    information_content.push_back(x.information_content);
    if (return_entropy) {
      entropy.push_back(compute_entropy(x.distribution));
    }
    if (return_distribution) {
      distribution.push_back(x.distribution);
    }
    if (decay) {
      pos.push_back(x.pos);
      time.push_back(x.time);
    }
  }
};

class model_order {
public:
  int chosen;
  int longest_available;
  bool deterministic_any;
  int deterministic_shortest;
  bool deterministic_is_selected;
  
  model_order(int chosen_,
              int longest_available_,
              bool deterministic_any_,
              int deterministic_shortest_, 
              bool deterministic_is_selected_) {
    chosen = chosen_;
    longest_available = longest_available_;
    deterministic_any = deterministic_any_;
    deterministic_shortest = deterministic_shortest_;
    deterministic_is_selected = deterministic_is_selected_;
  }
};

class ppm {
public:
  int alphabet_size;
  int order_bound;
  bool shortest_deterministic;
  bool exclusion;
  bool update_exclusion;
  std::string escape;
  double k;
  bool decay;
  bool sub_n_from_m1_dist;
  bool lambda_uses_zero_weight_symbols;
  bool debug_smooth;
  std::vector<std::string> alphabet_levels;
  
  int num_observations = 0;
  std::vector<double> all_time;
  
  ppm(int alphabet_size_,
      int order_bound_,
      bool shortest_deterministic_,
      bool exclusion_,
      bool update_exclusion_,
      std::string escape_,
      bool decay_,
      bool sub_n_from_m1_dist_,
      bool lambda_uses_zero_weight_symbols_,
      std::vector<std::string> alphabet_levels_
  ) {
    if (alphabet_size_ <= 0) {
      throw std::invalid_argument("alphabet size must be greater than 0");
    }
    
    alphabet_size = alphabet_size_;
    order_bound = order_bound_;
    shortest_deterministic = shortest_deterministic_;
    exclusion = exclusion_;
    update_exclusion = update_exclusion_;
    escape = escape_;
    k = this->get_k(escape);
    decay = decay_;
    sub_n_from_m1_dist = sub_n_from_m1_dist_;
    lambda_uses_zero_weight_symbols = lambda_uses_zero_weight_symbols_;
    alphabet_levels = alphabet_levels_;
  }
  
  virtual ~ ppm() {};
  
  // returns true if the n_gram already existed in the memory bank
  virtual bool insert(sequence x, int pos, double time, bool full_only) {
    throw std::runtime_error("this shouldn't happen (1)");
    return true;
  };
  
  virtual double get_weight(const sequence &n_gram, 
                            int pos, 
                            double time, 
                            bool update_excluded) {
    return 0.0;
  };
  
  double get_num_observed_symbols(int pos, double time) {
    int res = 0;
    for (int i = 0; i < this->alphabet_size; i ++) {
      sequence symbol(1, i);
      double weight = get_weight(symbol, pos, time, false);
      if (weight > 0.0) res ++;
    }
    return res;
  }
  
  double get_context_count(const std::vector<double> &counts, 
                           const std::vector<bool> &excluded) {
    double context_count = 0;
    for (int i = 0; i < this->alphabet_size; i ++) {
      if (!excluded[i]) {
        context_count += counts[i];
      }
    }
    return context_count; 
  }
  
  sequence_prediction model_seq(sequence x,
                                std::vector<double> time = {},
                                bool train = true,
                                bool predict = true,
                                bool return_distribution = true,
                                bool return_entropy = true,
                                bool generate = false) {
    int n = x.size();
    if (this->decay && 
        (static_cast<unsigned int>(x.size()) != 
        static_cast<unsigned int>(time.size()))) {
      throw std::invalid_argument("time must either have length 0 or have length equal to x");
    }
    if (this->all_time.size() > 0 && time.size() > 0 && time[0] < this->all_time.back()) {
      throw std::range_error("a sequence may not begin before the previous sequence finished");
    }
    
    sequence_prediction result(return_distribution,
                               return_entropy,
                               this->decay);
    
    for (int i = 0; i < n; i ++) {
      int pos_i = num_observations;
      double time_i = this->decay? time[i] : 0;
      // Predict
      if (predict) {
        sequence context = (i < 1 || order_bound < 1) ? sequence() :
        subseq(x,
               std::max(0, i - order_bound),
               i - 1);
        symbol_prediction pred = predict_symbol(x[i], context, pos_i, time_i, generate);
        result.insert(pred);
        if (generate) {
          x[i] = pred.symbol;
        }
      }
      // Train
      if (train) {
        if (decay) this->all_time.push_back(time_i);
        bool full_only = false;
        for (int h = std::max(0, i - order_bound); h <= i; h ++) {
          full_only = this->insert(subseq(x, h, i), pos_i, time_i, full_only);
        }
        num_observations ++;
      }
    }
    return(result);
  }
  
  symbol_prediction predict_symbol(
      int symbol, 
      const sequence &context,
      int pos, 
      double time,
      bool generate
    ) {
    if (!generate) { 
      if (symbol < 0) {
        throw std::invalid_argument("symbols must be greater than or equal to 0");
      }
      if (symbol > alphabet_size - 1) {
        throw std::invalid_argument("symbols cannot exceed (alphabet_size - 1)");
      }
    }
    
    model_order model_order = this->get_model_order(context, pos, time);
    std::vector<double> dist = get_probability_distribution(context,
                                                            model_order,
                                                            pos,
                                                            time);
    
    if (generate) {
      std::random_device rd;
      std::mt19937 gen(rd());
      std::discrete_distribution<> d(dist.begin(), dist.end());
      symbol = d(gen);
    }
  
    symbol_prediction out(symbol, pos, time, model_order.chosen, dist);
    return(out);
  } 
  
  std::vector<double> get_probability_distribution(const sequence &context,
                                                   model_order model_order,
                                                   int pos, 
                                                   double time) {
    std::vector<bool> excluded(alphabet_size, false);
    std::vector<double> dist = get_smoothed_distribution(
      context,
      model_order,
      model_order.chosen,
      pos,
      time,
      excluded
    );
    return normalise_distribution(dist);
  }
  
  std::vector<double> get_smoothed_distribution(const sequence &context,
                                                model_order model_order, 
                                                int order,
                                                int pos, 
                                                double time,
                                                std::vector<bool> &excluded) {
    if (order == -1) {
      return get_order_minus_1_distribution(pos, time);
    } else {
      bool update_excluded = this->update_exclusion;
      if (order == model_order.chosen &&
          this->shortest_deterministic &&
          this->update_exclusion &&
          model_order.deterministic_is_selected) {
        update_excluded = false;
      }
      
      std::vector<int> n_gram = last_n(context, order);
      n_gram.resize(order + 1);
      
      std::vector<double> counts(this->alphabet_size);
      int num_distinct_symbols = 0;
      std::vector<bool> predicted(this->alphabet_size);
      
      for (int i = 0; i < this->alphabet_size; i ++) {
        n_gram[order] = i;
        counts[i] = this->get_weight(n_gram, pos, time, update_excluded);
        if (counts[i] > 0.0) {
          predicted[i] = true;
          num_distinct_symbols += 1;
        } else {
          predicted[i] = false;
        }
        counts[i] = this->modify_count(counts[i]);
      }
      
      // std::cout << "counts: ";
      // print(counts);
      
      double context_count = get_context_count(counts, excluded);
      double lambda = get_lambda(counts, context_count, num_distinct_symbols);
      
      std::vector<double> alphas = get_alphas(lambda, counts, context_count);
      
      if (this->debug_smooth) {
        std::cout << "\n*** order = " << order << " ***\n";
        std::cout << "pos = " << pos << "\n";
        std::cout << "time = " << time << "\n";
        std::cout << "model_order.chosen = " << model_order.chosen << "\n";
        // std::cout << "this->shortest_deterministic = " << this->shortest_deterministic << "\n";
        // std::cout << "this->update_exclusion = " << this->update_exclusion << "\n";
        // std::cout << "model_order.deterministic_is_selected = " << model_order.deterministic_is_selected << "\n";
        std::cout << "context = ";
        print(last_n(context, order));
        // std::cout << "update_excluded = " << update_excluded << "\n";
        std::cout << "counts = ";
        print(counts);
        std::cout << "context_count = " << context_count << "\n";
        std::cout << "lambda = " << lambda << "\n";
        std::cout << "alphas = ";
        print(alphas);
      }
      
      if (this->exclusion) {
        for (int i = 0; i < alphabet_size; i ++) {
          // There is a choice here:
          // do we exclude symbols that have alphas greater than 0
          // (i.e. their counts survive addition of k),
          // or do we exclude any symbol that is present in the tree at all,
          // even if adding k takes it down to 0?
          //
          // Since decay-based models don't have exclusion,
          // we only have to think about normal PPM models.
          // All of these models apart from PPM-B have k > -1,
          // in which case there is no difference between the strategies.
          // We only have to worry for PPM-B.
          //
          // Following Bunton (1996) and Pearce (2005)'s implementation,
          // we adopt the latter strategy, excluding symbols even 
          // if their alphas are equal to 0, as long as they were present
          // in the tree.
          
          if (predicted[i]) {
            excluded[i] = true;
          }
        }
        if (this->debug_smooth) {
          std::cout << "new excluded = ";
          print(excluded);
        }
      }
      
      std::vector<double> lower_order_distribution = get_smoothed_distribution(
        context, model_order, order - 1, pos, time, excluded);
      
      std::vector<double> res(this->alphabet_size);
      for (int i = 0; i < this->alphabet_size; i ++) {
        res[i] = alphas[i] + (1 - lambda) * lower_order_distribution[i];
      }
      
      if (this->debug_smooth) {
        std::cout << "order " << order << " ";
        std::cout << "probability distribution = ";
        print(res);
      }
      
      return res;
    }
  }
  
  std::vector<double>get_alphas(double lambda, 
                                const std::vector<double> &counts, 
                                double context_count) {
    if (lambda > 0) {
      std::vector<double> res(this->alphabet_size);
      for (int i = 0; i < this->alphabet_size; i ++) {
        res[i] = lambda * counts[i] / context_count;
      }
      return res;
    } else {
      std::vector<double> res(this->alphabet_size, 0);
      return res;
    }
  }
  
  // The need to capture situations where the context_count is 0 is
  // introduced by Pearce (2005)'s decision to introduce exclusion
  // (see 6.2.3.3), though the thesis does not mention
  // this explicitly.
  virtual double get_lambda(const std::vector<double> &counts, double context_count, int num_distinct_symbols) {
    throw std::runtime_error("this virtual get_lambda method should never be called directly");
    return 0.0;
  }
  
  double get_k(const std::string &e) {
    if (e == "a") {
      return 0;
    } else if (e == "b") {
      return - 1;
    } else if (e == "c") {
      return 0;
    } else if (e == "d") {
      return - 0.5;
    } else if (e == "ax") {
      return 0;
    } else {
      throw std::invalid_argument("unrecognised escape method");
    }
  }
  
  double get_effective_distinct_symbols(int num_distinct_symbols, 
                                        const std::vector<double> &counts) {
    if (this->lambda_uses_zero_weight_symbols) {
      return num_distinct_symbols;
    } else {
      return this->count_positive_values(counts);
    }
  }
  
  double lambda_a(const std::vector<double> &counts, double context_count, int num_distinct_symbols) {
    if (this->debug_smooth) {
      std::cout << "lambda_a, context_count = " << context_count << "\n";
    }
    return context_count / (context_count + 1.0);
  }
  
  double lambda_b(const std::vector<double> &counts, double context_count, int num_distinct_symbols) {
    double effective_distinct_symbols = 
      this->get_effective_distinct_symbols(num_distinct_symbols,
                                           counts);
    
    return static_cast<double>(context_count) /
      static_cast<double>(context_count + effective_distinct_symbols);
  }
  
  double lambda_c(const std::vector<double> &counts, double context_count, int num_distinct_symbols) {
    double effective_distinct_symbols = 
      this->get_effective_distinct_symbols(num_distinct_symbols,
                                           counts);
    
    return static_cast<double>(context_count) /
      static_cast<double>(context_count + effective_distinct_symbols);
  }
  
  double lambda_d(const std::vector<double> &counts, double context_count, int num_distinct_symbols) {
    double effective_distinct_symbols = 
      this->get_effective_distinct_symbols(num_distinct_symbols,
                                           counts);
    
    return static_cast<double>(context_count) /
      (static_cast<double>(context_count + effective_distinct_symbols / 2.0));
  }
  
  double lambda_ax(const std::vector<double> &counts, double context_count, int num_distinct_symbols) {
    // Note - there is a mistake in the reference papers, 
    // Pearce & Wiggins (2004), also Pearce (2005);
    // the 1.0 is missing from the equation.
    // Our version is consistent with the context literature though,
    // and consistent with Pearce's LISP implementation.
    //
    // We generalise the definition of singletons to decayed counts between
    // 0 and 1. This is a bit hacky though, and the escape method
    // should ultimately be reconfigured for new decay functions.
    return static_cast<double>(context_count) /
      static_cast<double>(context_count + num_singletons(counts) + 1.0);
  }
  
  int num_singletons(const std::vector<double> &x) {
    int n = static_cast<int>(x.size());
    int res = 0;
    for (int i = 0; i < n; i ++) {
      if (x[i] > 0 && x[i] <= 1) {
        res ++;
      }
    }
    return res;
  }
  
  double modify_count(double count) {
    if (this->k == 0 || count == 0) {
      return count;
    } else {
      double x = count + this->k;
      if (x > 0) {
        return x;
      } else {
        return 0;
      }
    }
  }
  
  int count_positive_values(const std::vector<double> &x) {
    int n = static_cast<int>(x.size());
    int res = 0;
    for (int i = 0; i < n; i ++) {
      if (x[i] > 0) {
        res ++;
      }
    }
    return res;
  }
  
  std::vector<double> get_order_minus_1_distribution(int pos, double time) {
    
    // See Bunton (1996, p. 82): alpha(s0) comes from the 3-arg version of count(),
    // which does not include exclusion or subtraction 
    // of the k parameter (see escape method).
    // It instead corresponds to the number of symbols that the model 
    // has ever seen.
    
    //// Old version:
    // int num_observed_symbols = 0;
    // for (int i = 0; i < this->alphabet_size; i ++) {
    //   if (excluded[i]) {
    //     num_observed_symbols ++;
    //   } 
    // }
    
    double denominator = this->alphabet_size + 1;
    
    if (this->sub_n_from_m1_dist) {
      // This is disabled for decay-based models
      double num_observed_symbols = this->get_num_observed_symbols(pos, time);
      denominator -= num_observed_symbols;
    }
    
    double p = 1.0 / denominator;
    std::vector<double> res(this->alphabet_size, p);
    
    if (this->debug_smooth) {
      std::cout << "order minus 1 distribution = ";
      print(res);
      std::cout << "\n";
    }
    return res;
  }
  
  model_order get_model_order(const sequence &context, int pos, double time) {
    const int longest_available = this->get_longest_context(context, pos, time);
    int chosen = longest_available;
    
    int det_shortest = - 1;
    int det_any = false;
    int det_is_selected = false;
    
    if (shortest_deterministic) {
      int det_shortest = this->get_shortest_deterministic_context(context,
                                                                  pos,
                                                                  time);
      bool det_any = det_shortest >= 0;
      if (det_any) {
        if (det_shortest < longest_available) {
          det_is_selected = true;
          chosen = det_shortest;
        }
      }
    }
    
    return(model_order(chosen, longest_available,
                       det_any, det_shortest, det_is_selected));
  }
  
  virtual int get_longest_context(sequence context, int pos, double time) {
    throw std::runtime_error("this shouldn't happen (2)");
    return 0;
  }
  
  int get_shortest_deterministic_context(const sequence &context, int pos, double time) {
    int len = static_cast<int>(context.size());
    int res = -1;
    for (int order = 0; order <= std::min(len, order_bound); order ++) {
      sequence effective_context = order == 0 ? sequence() : subseq(context, 
                                                         len - order, len - 1);
      if (is_deterministic_context(effective_context, pos, time)) {
        res = order;
        break;
      }
    }
    return(res);
  }
  
  bool is_deterministic_context(const sequence &context, int pos, double time) {
    int num_continuations = 0;
    for (int i = 0; i < alphabet_size; i ++) {
      sequence n_gram = context;
      n_gram.push_back(i);
      double weight = this->get_weight(n_gram, 
                                       pos, 
                                       time, 
                                       false); // update exclusion
      if (weight > 0) {
        num_continuations ++;
        if (num_continuations > 1) {
          break;
        }
      }
    }
    return num_continuations == 1;
  }
};

class ppm_simple: public ppm {
public:
  std::unordered_map<sequence, 
                     record_simple,
                     container_hash<sequence>> data;
  
  ppm_simple(
    int alphabet_size_,
    int order_bound_,
    bool shortest_deterministic_,
    bool exclusion_,
    bool update_exclusion_,
    std::string escape_,
    std::vector<std::string> alphabet_levels_
  ) : ppm(
      alphabet_size_,
      order_bound_, 
      shortest_deterministic_, 
      exclusion_, 
      update_exclusion_, 
      escape_,
      false, // decay
      true, // sub_n_from_m1_dist
      true, // lambda_uses_zero_weight_symbols
      alphabet_levels_
      ) { 
    data = {};
  }
  
  ~ ppm_simple() {};
  
  bool insert(sequence x, int pos, double time, bool full_only) {
    std::unordered_map<sequence, record_simple, container_hash<sequence>>::const_iterator target = data.find(x);
    if (target == data.end()) {
      record_simple record;
      record.add_1(full_only);
      data[x] = record;
      return false;
    } else {
      data[x].add_1(full_only);
      return true;
    }
  }
  
  int get_longest_context(sequence context, int pos, double time) {
    // std::cout << "get_longest_context...\n";
    int context_len = static_cast<int>(context.size());
    int upper_bound = std::min(order_bound, context_len);
    
    for (int order = upper_bound; order >= 0; order --) {
      // std::cout << "Checking order = " << order << "\n";
      sequence x = order == 0 ? sequence() : subseq(context,
                                         context_len - order,
                                         context_len - 1);
      // std::cout << "Truncated context = ";
      // print(x);
      // Skip this iteration if the context doesn't exist in the tree
      if (order > 0 && // we don't store 0-grams in the tree
          this->get_weight(x, 
                           0, // pos - irrelevant for non-decay-based models
                           0, // time - irrelevant for non-decay-based models
                           false) // update exclusion
            == 0.0) {
        // std::cout << "Couldn't find context in the tree\n";
        continue;
      }
      // Skip this iteration if we can't find a continuation for that context
      bool any_continuation = false;
      x.resize(order + 1);
      for (int i = 0; i < this->alphabet_size; i ++) {
        x[order] = i;
        if (this->get_weight(x, 0, 0, false) > 0.0) {
          any_continuation = true;
          break;
        }
      }
      if (! any_continuation) {
        // std::cout << "Couldn't find any continuations for this context\n";
        continue;
      }
      // std::cout << "Couldn't find a problem with this context\n";
      return(order);
    }
    // std::cout << "Escaped to order = -1\n";
    return(- 1);
  }
  
  
  double get_weight(const sequence &n_gram, 
                    int pos, 
                    double time,
                    bool update_excluded) {
    return static_cast<double>(this->get_count(n_gram, update_excluded));
  };
  
  long int get_count(const sequence &x, bool update_excluded) {
    std::unordered_map<sequence, record_simple, container_hash<sequence>>::const_iterator target = data.find(x);
    if (target == data.end()) {
      return(0);
    } else if (update_excluded) {
      return target->second.up_ex_count;
    } else {
      return target->second.full_count;
    }
  }
  
  // The need to capture situations where the context_count is 0 is
  // introduced by Pearce (2005)'s decision to introduce exclusion
  // (see 6.2.3.3), though the thesis does not mention
  // this explicitly.
  double get_lambda(const std::vector<double> &counts, double context_count, int num_distinct_symbols) {
    if (this->debug_smooth) {
      std::cout << "calling ppm_simple.get_lambda()\n";
    }
    std::string e = this->escape;
    if (context_count <= 0.0) {
      return 0.0;
    } else if (e == "a") {
      return this->lambda_a(counts, context_count, num_distinct_symbols);
    } else if (e == "b") {
      return this->lambda_b(counts, context_count, num_distinct_symbols);
    } else if (e == "c") {
      return this->lambda_c(counts, context_count, num_distinct_symbols);
    } else if (e == "d") {
      return this->lambda_d(counts, context_count, num_distinct_symbols);
    } else if (e == "ax") {
      return this->lambda_ax(counts, context_count, num_distinct_symbols);
    } else {
      throw std::invalid_argument("unrecognised escape method");
    }
  }
};


NB_MODULE(ppm, m) {
  nb::class_<ppm>(m, "ppm")
    // ppm constructor not made available in NB
    .def("model_seq", &ppm::model_seq)
    .def("get_weight", &ppm::get_weight)
    .def_rw("alphabet_size", &ppm::alphabet_size)
    .def_rw("order_bound", &ppm::order_bound)
    .def_rw("shortest_deterministic", &ppm::shortest_deterministic)
    .def_rw("exclusion", &ppm::exclusion)
    .def_rw("update_exclusion", &ppm::update_exclusion)
    .def_rw("escape", &ppm::escape)
    .def_rw("all_time", &ppm::all_time)
    .def_rw("sub_n_from_m1_dist", &ppm::sub_n_from_m1_dist)
    .def_rw("lambda_uses_zero_weight_symbols", &ppm::lambda_uses_zero_weight_symbols)
    .def_rw("debug_smooth", &ppm::debug_smooth)
    .def_rw("alphabet_levels", &ppm::alphabet_levels)
  ;

  nb::class_<ppm_simple, ppm>(m, "ppm_simple")
    .def(nb::init<int, int, bool, bool, bool, std::string, std::vector<std::string>>())
    .def("get_count", &ppm_simple::get_count)
  ;

  nb::class_<sequence_prediction>(m, "sequence_prediction")
    .def_ro("symbol", &sequence_prediction::symbol)
    .def_ro("pos", &sequence_prediction::pos)
    .def_ro("time", &sequence_prediction::time)
    .def_ro("model_order", &sequence_prediction::model_order)
    .def_ro("information_content", &sequence_prediction::information_content)
    .def_ro("entropy", &sequence_prediction::entropy)
    .def_ro("distribution", &sequence_prediction::distribution)
  ;
}
