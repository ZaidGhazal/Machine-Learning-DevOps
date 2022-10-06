"""Logging hands-on"""
## STEPS TO COMPLETE ##
# 1. import logging
# 2. set up config file for logging called `results.log`
# 3. add try except with logging for success or error
#    in relation to checking the types of a and b
# 4. check to see that log is created and populated correctly
#    should have error for first function and success for
#    the second
# 5. use pylint and autopep8 to make changes
#    and receive 10/10 pep8 score
import logging

logging.basicConfig(
    filename='results.log',
    level=logging.INFO,
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s')


def sum_vals(num1, num2) -> int:
    '''
    Args:
        a: (int)
        b: (int)
    Return:
        a + b (int)
    '''
    try:
        summ = int(num1) + int(num2)
        return summ
    except ValueError as err:
        logging.error("ValueError: %s", str(err))
        return None


if __name__ == "__main__":
    sum_vals('no', 'way')
    sum_vals(4, 5)
