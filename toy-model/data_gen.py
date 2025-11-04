import random
import os
import csv
import math

class DataGenerator:
    def __init__(self):
        self.TRAIN_PATH = "data/train/"
        self.VAL_PATH = "data/val/"
        os.makedirs(self.TRAIN_PATH, exist_ok=True)
        os.makedirs(self.VAL_PATH, exist_ok=True)
        self._MAX_N = (1 << 64) - 1

    # ---------------------------
    # Internal helpers
    # ---------------------------
    _MAX_N = (1 << 64) - 1

    @staticmethod
    def _rand_u64():
        return random.getrandbits(64)

    @staticmethod
    def _write_dataset(all_samples, file_stem, train_split=0.8):
        random.shuffle(all_samples)
        split_idx = int(len(all_samples) * train_split)
        train_samples = all_samples[:split_idx]
        val_samples = all_samples[split_idx:]

        train_file_path = os.path.join("data/train/", f"{file_stem}.csv")
        val_file_path = os.path.join("data/val/", f"{file_stem}.csv")

        with open(train_file_path, 'w', newline='') as train_f:
            train_writer = csv.writer(train_f)
            train_writer.writerow(['number', 'label'])
            train_writer.writerows(train_samples)

        with open(val_file_path, 'w', newline='') as val_f:
            val_writer = csv.writer(val_f)
            val_writer.writerow(['number', 'label'])
            val_writer.writerows(val_samples)

    @staticmethod
    def _popcount(n: int) -> int:
        return n.bit_count()

    @staticmethod
    def _leading_zeros(n: int) -> int:
        if n == 0:
            return 64
        return 64 - n.bit_length()

    @staticmethod
    def _trailing_zeros(n: int) -> int:
        if n == 0:
            return 64
        return (n & -n).bit_length() - 1

    @staticmethod
    def _is_power_of_two(x: int) -> bool:
        return x > 0 and (x & (x - 1)) == 0

    @staticmethod
    def _longest_run_of_ones(n: int) -> int:
        max_run = 0
        cur = 0
        while n:
            if n & 1:
                cur += 1
                if cur > max_run:
                    max_run = cur
            else:
                cur = 0
            n >>= 1
        return max_run

    @staticmethod
    def _is_perfect_square(x: int) -> bool:
        if x < 0:
            return False
        s = math.isqrt(x)
        return s * s == x

    @staticmethod
    def _int_nth_root(n: int, k: int) -> int:
        # floor of the k-th root using binary search
        if n < 2:
            return n
        low, high = 1, 1 << (64 // k + 2)
        # tighten upper bound
        while pow(high, k) <= n:
            high <<= 1
        low = high >> 1
        while low < high:
            mid = (low + high + 1) // 2
            p = pow(mid, k)
            if p == n:
                return mid
            if p < n:
                low = mid
            else:
                high = mid - 1
        return low

    @staticmethod
    def _is_perfect_k_power(n: int, k: int) -> bool:
        if k < 2:
            return False
        r = DataGenerator._int_nth_root(n, k)
        return pow(r, k) == n

    # Miller-Rabin deterministic for 64-bit
    @staticmethod
    def _is_prime_64(n: int) -> bool:
        if n < 2:
            return False
        # small primes
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for p in small_primes:
            if n % p == 0:
                return n == p
        # write n-1 = d*2^s
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
        # set of bases sufficient for 64-bit determinism
        for a in [2, 3, 5, 7, 11, 13, 17]:
            if a % n == 0:
                continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            skip_to_next_n = False
            for _ in range(s - 1):
                x = pow(x, 2, n)
                if x == n - 1:
                    skip_to_next_n = True
                    break
            if skip_to_next_n:
                continue
            return False
        return True

    @staticmethod
    def divisible_by_k(k, num_samples, train_split=0.8):
        if k <= 0:
            raise ValueError("k must be positive integer > 0")

        train_file_path = os.path.join("data/train/", f"div_by_{k}.csv")
        val_file_path = os.path.join("data/val/", f"div_by_{k}.csv")

        # Generate equal positives and negatives
        max_n = (1 << 64) - 1  # 2^64 - 1
        positives = []
        negatives = []
        
        # Positives: random multiples within range
        max_multiple = max_n // k
        for _ in range(math.ceil(num_samples / 2)):
            multiple = random.randint(0, max_multiple)
            positives.append((multiple * k, 1))
        
        # Negatives: random non-multiples
        for _ in range(math.ceil(num_samples / 2)):
            n = random.randint(0, max_n)
            while n % k == 0:
                n = random.randint(0, max_n)
            negatives.append((n, 0))
        
        # Combine, shuffle, split
        all_samples = positives + negatives
        random.shuffle(all_samples)
        split_idx = int(len(all_samples) * train_split)
        train_samples = all_samples[:split_idx]
        val_samples = all_samples[split_idx:]
        
        # Write with headers
        with open(train_file_path, 'w', newline='') as train_f:
            train_writer = csv.writer(train_f)
            train_writer.writerow(['number', 'label'])
            train_writer.writerows(train_samples)
        
        with open(val_file_path, 'w', newline='') as val_f:
            val_writer = csv.writer(val_f)
            val_writer.writerow(['number', 'label'])
            val_writer.writerows(val_samples)

    # ---------------------------
    # Additional dataset generators
    # ---------------------------

    @staticmethod
    def mod_equals(k: int, r: int, num_samples: int, train_split: float = 0.8):
        if k <= 0:
            raise ValueError("k must be positive integer > 0")
        if r < 0 or r >= k:
            raise ValueError("r must satisfy 0 <= r < k")
        max_n = DataGenerator._MAX_N
        positives, negatives = [], []

        # Positives: n = m*k + r, within [0, max_n]
        max_m = (max_n - r) // k
        for _ in range(math.ceil(num_samples / 2)):
            m = random.randint(0, max_m)
            positives.append((m * k + r, 1))

        # Negatives: n % k != r
        for _ in range(math.ceil(num_samples / 2)):
            n = random.randint(0, max_n)
            while n % k == r:
                n = random.randint(0, max_n)
            negatives.append((n, 0))

        DataGenerator._write_dataset(positives + negatives, f"mod_{k}_eq_{r}", train_split)

    @staticmethod
    def trailing_zeros_ge(t: int, num_samples: int, train_split: float = 0.8):
        if t < 0 or t > 64:
            raise ValueError("t must satisfy 0 <= t <= 64")
        positives, negatives = [], []
        max_n = (1 << 64) - 1
        if t == 64:
            for _ in range(math.ceil(num_samples / 2)):
                positives.append((0, 1))
        else:
            step = 1 << t
            max_m = max_n // step
            for _ in range(math.ceil(num_samples / 2)):
                m = random.randint(0, max_m)
                positives.append((m * step, 1))
        for _ in range(math.ceil(num_samples / 2)):
            n = random.randint(0, max_n)
            while DataGenerator._trailing_zeros(n) >= t:
                n = random.randint(0, max_n)
            negatives.append((n, 0))
        DataGenerator._write_dataset(positives + negatives, f"tz_ge_{t}", train_split)

    @staticmethod
    def popcount_mod_equals(m: int, s: int, num_samples: int, train_split: float = 0.8):
        if m <= 0:
            raise ValueError("m must be positive integer > 0")
        if s < 0 or s >= m:
            raise ValueError("s must satisfy 0 <= s < m")
        positives, negatives = [], []
        max_n = DataGenerator._MAX_N
        # Positives
        for _ in range(math.ceil(num_samples / 2)):
            n = random.randint(0, max_n)
            while DataGenerator._popcount(n) % m != s:
                n = random.randint(0, max_n)
            positives.append((n, 1))
        # Negatives
        for _ in range(math.ceil(num_samples / 2)):
            n = random.randint(0, max_n)
            while DataGenerator._popcount(n) % m == s:
                n = random.randint(0, max_n)
            negatives.append((n, 0))
        DataGenerator._write_dataset(positives + negatives, f"popcount_mod_{m}_eq_{s}", train_split)

    @staticmethod
    def exact_popcount(p: int, num_samples: int, train_split: float = 0.8):
        if p < 0 or p > 64:
            raise ValueError("p must satisfy 0 <= p <= 64")
        positives, negatives = [], []
        # Positives: choose exactly p positions out of 64
        for _ in range(math.ceil(num_samples / 2)):
            if p == 0:
                n = 0
            elif p == 64:
                n = (1 << 64) - 1
            else:
                positions = random.sample(range(64), p)
                n = 0
                for pos in positions:
                    n |= (1 << pos)
            positives.append((n, 1))
        # Negatives
        max_n = DataGenerator._MAX_N
        for _ in range(math.ceil(num_samples / 2)):
            n = random.randint(0, max_n)
            while DataGenerator._popcount(n) == p:
                n = random.randint(0, max_n)
            negatives.append((n, 0))
        DataGenerator._write_dataset(positives + negatives, f"popcount_eq_{p}", train_split)

    @staticmethod
    def binary_palindrome(num_samples: int, train_split: float = 0.8):
        positives, negatives = [], []
        max_n = DataGenerator._MAX_N

        def make_pal(L: int) -> int:
            # L >= 1, MSB must be 1 so representation has no leading zeros
            half = (L + 1) // 2
            # ensure first bit is 1
            bits = [1] + [random.randint(0, 1) for _ in range(half - 1)]
            # mirror
            if L % 2 == 0:
                mirrored = bits + bits[::-1]
            else:
                mirrored = bits + bits[:-1][::-1]
            n = 0
            for b in mirrored:
                n = (n << 1) | b
            return n

        # Positives
        for _ in range(math.ceil(num_samples / 2)):
            L = random.randint(1, 64)
            n = make_pal(L)
            positives.append((n, 1))

        # Negatives: ensure not palindrome (by flipping a non-symmetric bit if needed)
        for _ in range(math.ceil(num_samples / 2)):
            n = random.randint(0, max_n)
            s = bin(n)[2:]
            if s == s[::-1]:
                # flip bit 0 (LSB) to break palindrome while staying in range
                n ^= 1
            negatives.append((n, 0))
        DataGenerator._write_dataset(positives + negatives, "bin_palindrome", train_split)

    @staticmethod
    def highest_set_bit_eq(h: int, num_samples: int, train_split: float = 0.8):
        if h < 0 or h > 63:
            raise ValueError("h must satisfy 0 <= h <= 63")
        positives, negatives = [], []
        max_n = DataGenerator._MAX_N
        # Positives: bit h set, bits above h zero
        for _ in range(math.ceil(num_samples / 2)):
            high_part = 0  # ensure no bits above h
            low_mask = (1 << h) - 1
            low_bits = random.getrandbits(h) if h > 0 else 0
            n = (1 << h) | low_bits
            positives.append((n, 1))
        # Negatives
        for _ in range(math.ceil(num_samples / 2)):
            n = random.randint(0, max_n)
            while n != 0 and (n.bit_length() - 1) == h:
                n = random.randint(0, max_n)
            # if n == 0, highest set bit undefined -> acceptable negative
            negatives.append((n, 0))
        DataGenerator._write_dataset(positives + negatives, f"msb_eq_{h}", train_split)

    @staticmethod
    def lowest_set_bit_eq(l: int, num_samples: int, train_split: float = 0.8):
        if l < 0 or l > 63:
            raise ValueError("l must satisfy 0 <= l <= 63")
        positives, negatives = [], []
        max_n = DataGenerator._MAX_N
        # Positives: bit l set, lower bits 0, higher bits random
        for _ in range(math.ceil(num_samples / 2)):
            higher_bits = random.getrandbits(63 - l) if l < 63 else 0
            n = (higher_bits << (l + 1)) | (1 << l)
            positives.append((n, 1))
        # Negatives
        for _ in range(math.ceil(num_samples / 2)):
            n = random.randint(0, max_n)
            while n != 0 and DataGenerator._trailing_zeros(n) == l:
                n = random.randint(0, max_n)
            # n == 0 has no set bits, acceptable negative
            negatives.append((n, 0))
        DataGenerator._write_dataset(positives + negatives, f"lsb_eq_{l}", train_split)

    @staticmethod
    def perfect_kth_power(k: int, num_samples: int, train_split: float = 0.8):
        if k < 2:
            raise ValueError("k must be integer >= 2")
        positives, negatives = [], []
        max_n = DataGenerator._MAX_N
        # Positives: n = a^k
        # Find maximum a such that a^k <= max_n
        a_max = DataGenerator._int_nth_root(max_n, k)
        for _ in range(math.ceil(num_samples / 2)):
            a = random.randint(1, a_max)
            n = pow(a, k)
            positives.append((n, 1))
        # Negatives: sample non perfect k-th powers
        for _ in range(math.ceil(num_samples / 2)):
            n = random.randint(0, max_n)
            while DataGenerator._is_perfect_k_power(n, k):
                n = random.randint(0, max_n)
            negatives.append((n, 0))
        DataGenerator._write_dataset(positives + negatives, f"perfect_{k}th_power", train_split)

    @staticmethod
    def mersenne_number(num_samples: int, train_split: float = 0.8):
        positives, negatives = [], []
        max_n = DataGenerator._MAX_N
        # Positives: n = 2^k - 1, allow k in [1,64]
        for _ in range(math.ceil(num_samples / 2)):
            k = random.randint(1, 64)
            n = (1 << k) - 1
            positives.append((n, 1))
        # Negatives: ensure n+1 not a power of two
        for _ in range(math.ceil(num_samples / 2)):
            n = random.randint(0, max_n)
            while DataGenerator._is_power_of_two(n + 1):
                n = random.randint(0, max_n)
            negatives.append((n, 0))
        DataGenerator._write_dataset(positives + negatives, "mersenne", train_split)

    @staticmethod
    def lsb_all_ones(length: int, num_samples: int, train_split: float = 0.8):
        if length < 0 or length > 64:
            raise ValueError("length must satisfy 0 <= length <= 64")
        positives, negatives = [], []
        max_n = DataGenerator._MAX_N
        mask = (1 << length) - 1 if length > 0 else 0
        # Positives: low 'length' bits all 1
        for _ in range(math.ceil(num_samples / 2)):
            if length == 64:
                n = mask
            else:
                high_bits = random.getrandbits(64 - length) if length < 64 else 0
                n = (high_bits << length) | mask
            positives.append((n, 1))
        # Negatives
        for _ in range(math.ceil(num_samples / 2)):
            n = random.randint(0, max_n)
            while (n & mask) == mask:
                n = random.randint(0, max_n)
            negatives.append((n, 0))
        DataGenerator._write_dataset(positives + negatives, f"lsb_all_ones_len_{length}", train_split)

    @staticmethod
    def xor_of_all_bits_equals(v: int, num_samples: int, train_split: float = 0.8):
        if v not in (0, 1):
            raise ValueError("v must be 0 or 1")
        positives, negatives = [], []
        max_n = DataGenerator._MAX_N
        # Positives: adjust parity if needed
        for _ in range(math.ceil(num_samples / 2)):
            n = random.randint(0, max_n)
            if (DataGenerator._popcount(n) & 1) != v:
                # flip LSB to change parity
                n ^= 1
            positives.append((n, 1))
        # Negatives
        for _ in range(math.ceil(num_samples / 2)):
            n = random.randint(0, max_n)
            while (DataGenerator._popcount(n) & 1) == v:
                n = random.randint(0, max_n)
            negatives.append((n, 0))
        DataGenerator._write_dataset(positives + negatives, f"xor_eq_{v}", train_split)

    @staticmethod
    def upper_half_popcount_eq(p: int, num_samples: int, train_split: float = 0.8):
        if p < 0 or p > 32:
            raise ValueError("p must satisfy 0 <= p <= 32")
        positives, negatives = [], []
        max_n = DataGenerator._MAX_N
        # Positives: exactly p ones in bits [32..63]
        for _ in range(math.ceil(num_samples / 2)):
            if p == 0:
                upper = 0
            elif p == 32:
                upper = (1 << 32) - 1
            else:
                positions = random.sample(range(32), p)
                upper = 0
                for pos in positions:
                    upper |= (1 << pos)
            lower = random.getrandbits(32)
            n = (upper << 32) | lower
            positives.append((n, 1))
        # Negatives: any with different popcount in upper half
        for _ in range(math.ceil(num_samples / 2)):
            n = random.randint(0, max_n)
            while ((n >> 32) & ((1 << 32) - 1)).bit_count() == p:
                n = random.randint(0, max_n)
            negatives.append((n, 0))
        DataGenerator._write_dataset(positives + negatives, f"upper32_popcount_eq_{p}", train_split)

    @staticmethod
    def is_prime(num_samples: int, train_split: float = 0.8):
        positives, negatives = [], []
        max_n = DataGenerator._MAX_N
        # Positives: primes
        count = 0
        while count < math.ceil(num_samples / 2):
            n = random.randint(2, max_n)
            if DataGenerator._is_prime_64(n):
                positives.append((n, 1))
                count += 1
        # Negatives: non-primes (0,1, composite)
        count = 0
        while count < math.ceil(num_samples / 2):
            n = random.randint(0, max_n)
            if not DataGenerator._is_prime_64(n):
                negatives.append((n, 0))
                count += 1
        DataGenerator._write_dataset(positives + negatives, "prime", train_split)

    @staticmethod
    def is_fibonacci(num_samples: int, train_split: float = 0.8):
        positives, negatives = [], []
        max_n = DataGenerator._MAX_N

        # Precompute all Fibonacci numbers within 64-bit
        fibs = [0, 1]
        while True:
            nxt = fibs[-1] + fibs[-2]
            if nxt > max_n:
                break
            fibs.append(nxt)
        fib_set = set(fibs)

        # Positives: sample from fibs
        for _ in range(math.ceil(num_samples / 2)):
            n = random.choice(fibs)
            positives.append((n, 1))
        # Negatives: sample not in fibs
        for _ in range(math.ceil(num_samples / 2)):
            n = random.randint(0, max_n)
            while n in fib_set:
                n = random.randint(0, max_n)
            negatives.append((n, 0))
        DataGenerator._write_dataset(positives + negatives, "fibonacci", train_split)

    @staticmethod
    def leading_zeros_ge(lz: int, num_samples: int, train_split: float = 0.8):
        if lz <= 0 or lz > 64:
            raise ValueError("lz must be positive integer 1 <= lz <= 64")
        positives, negatives = [], []
        max_n = (1 << 64) - 1
        threshold = 1 << (64 - lz) if lz < 64 else 1
        for _ in range(math.ceil(num_samples / 2)):
            if lz == 64:
                n = 0
            else:
                n = random.randint(0, threshold - 1)
            positives.append((n, 1))
        for _ in range(math.ceil(num_samples / 2)):
            n = random.randint(threshold, max_n)
            negatives.append((n, 0))
        DataGenerator._write_dataset(positives + negatives, f"lz_ge_{lz}", train_split)

    @staticmethod
    def bit_i_set(i: int, num_samples: int, train_split: float = 0.8):
        if i < 0 or i > 63:
            raise ValueError("i must satisfy 0 <= i <= 63")
        positives, negatives = [], []
        max_n = DataGenerator._MAX_N
        # Positives: ensure bit i is set
        for _ in range(math.ceil(num_samples / 2)):
            hi = random.getrandbits(63 - i) if i < 63 else 0
            lo = random.getrandbits(i) if i > 0 else 0
            n = (hi << (i + 1)) | (1 << i) | lo
            positives.append((n, 1))
        # Negatives: ensure bit i is not set
        for _ in range(math.ceil(num_samples / 2)):
            n = random.randint(0, max_n)
            while ((n >> i) & 1) == 1:
                n = random.randint(0, max_n)
            negatives.append((n, 0))
        DataGenerator._write_dataset(positives + negatives, f"bit_{i}_set", train_split)

    @staticmethod
    def consecutive_ones_ge(c: int, num_samples: int, train_split: float = 0.8):
        if c < 1 or c > 64:
            raise ValueError("c must satisfy 1 <= c <= 64")
        positives, negatives = [], []
        max_n = DataGenerator._MAX_N
        # Positives: plant a run of c ones starting at random position
        for _ in range(math.ceil(num_samples / 2)):
            start = random.randint(0, 64 - c)
            run_mask = ((1 << c) - 1) << start
            high_bits = random.getrandbits(64)
            n = (high_bits | run_mask) & max_n
            positives.append((n, 1))
        # Negatives: ensure longest run < c
        for _ in range(math.ceil(num_samples / 2)):
            n = random.randint(0, max_n)
            while DataGenerator._longest_run_of_ones(n) >= c:
                n = random.randint(0, max_n)
            negatives.append((n, 0))
        DataGenerator._write_dataset(positives + negatives, f"consec1s_ge_{c}", train_split)
