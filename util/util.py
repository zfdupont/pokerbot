from typing import List
import operator
import functools

from models.card import Card
from util.lookup_table import LookupTables


def popcount(v):
    c = 0
    while v:
        v &= v - 1
        c += 1
    return c


def card_to_binary(card: Card):
    b_mask = 1 << (14 + card.rank)
    q_mask = LookupTables.primes[card.suit_index - 1] << 12
    r_mask = (card.rank - 2) << 8
    p_mask = LookupTables.primes[card.rank - 2]
    return b_mask | q_mask | r_mask | p_mask


def card_to_binary_lookup(card: Card):
    return LookupTables.card_to_binary[card.rank][card.suit_index]


def hand_value(hole_cards: List[Card], community: List[Card]) -> int:
    hand = hole_cards + community

    binhand = list(map(card_to_binary, hand))
    flush_prime = functools.reduce(operator.mul, [(card >> 12) & 0xF for card in binhand])
    flush_suit = False
    if flush_prime in LookupTables.prime_products_to_flush:
        flush_suit = LookupTables.prime_products_to_flush[flush_prime]

    odd_xor = functools.reduce(operator.xor, binhand)
    even_xor = (functools.reduce(operator.or_, binhand) >> 16) ^ odd_xor

    if flush_suit:
        if even_xor == 0:
            bits = functools.reduce(operator.or_, [
                card >> 16 for card in binhand if (card >> 12) & 0xF == flush_suit
            ])
            return LookupTables.flush_rank_bits_to_rank[bits]
        else:
            if popcount(even_xor) == 2:
                return LookupTables.flush_rank_bits_to_rank[odd_xor | even_xor]
            else:
                bits = functools.reduce(operator.or_, [
                    card >> 16 for card in binhand if (card >> 12) & 0xF == flush_suit
                ])
                return LookupTables.flush_rank_bits_to_rank[bits]

    if even_xor == 0:
        odd_popcount = popcount(odd_xor)
        if odd_popcount == 7:
            return LookupTables.odd_xors_to_rank[odd_xor]
        else:
            prime_product = functools.reduce(operator.mul, [card & 0xFF for card in binhand])
            return LookupTables.prime_products_to_rank[prime_product]
    else:
        odd_popcount = popcount(odd_xor)
        if odd_popcount == 5:
            return LookupTables.even_xors_to_odd_xors_to_rank[even_xor][odd_xor]
        elif odd_popcount == 3:
            even_popcount = popcount(even_xor)
            if even_popcount == 2:
                return LookupTables.even_xors_to_odd_xors_to_rank[even_xor][odd_xor]
            else:
                prime_product = functools.reduce(operator.mul, [card & 0xFF for card in binhand])
                return LookupTables.prime_products_to_rank[prime_product]
        else:
            even_popcount = popcount(even_xor)
            if even_popcount == 3:
                return LookupTables.even_xors_to_odd_xors_to_rank[even_xor][odd_xor]
            elif even_popcount == 2:
                prime_product = functools.reduce(operator.mul, [card & 0xFF for card in binhand])
                return LookupTables.prime_products_to_rank[prime_product]
            else:
                return LookupTables.even_xors_to_odd_xors_to_rank[even_xor][odd_xor]
