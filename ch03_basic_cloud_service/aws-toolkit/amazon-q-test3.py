def prime_factorization(n: int) -> list[int]:
    """
    주어진 수의 소인수분해를 수행하는 함수
    """
    def is_prime(num: int) -> bool:
        if num < 2:
            return False
        for i in range(2, int(num**0.5) + 1):
            if num % i == 0:
                return False
        return True

    factors = []
    for i in range(2, n + 1):
        while n % i == 0 and is_prime(i):
            factors.append(i)
            n //= i
        if n == 1:
            break
    
    return factors

number = 12345678910111
print(f"{number}의 소인수: {prime_factorization(number)}")
# 이 계산은 큰 수에 대해 매우 오래 걸릴 것입니다.