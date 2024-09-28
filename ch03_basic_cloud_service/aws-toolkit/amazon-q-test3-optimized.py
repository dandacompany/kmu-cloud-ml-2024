def prime_factorization(n: int) -> list[int]:
    """
    주어진 수의 소인수분해를 수행하는 함수
    """
    def is_prime(num: int, primes: set[int]) -> bool:
        if num < 2:
            return False
        if num in primes:
            return True
        for i in range(2, int(num**0.5) + 1):
            if num % i == 0:
                return False
        primes.add(num)
        return True

    factors = []
    primes = set()
    for i in range(2, int(n**0.5) + 1):
        while n % i == 0 and is_prime(i, primes):
            factors.append(i)
            n //= i
        if n == 1:
            break
    if n > 1:
        factors.append(n)

    return factors

number = 12345678910111
print(f"{number}의 소인수: {prime_factorization(number)}")
