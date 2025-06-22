# Most of this work was migrated from https://github.com/ishaanam/wallet-fingerprinting/blob/master/change_heuristic.py

def get_spending_types(tx):
    types = []
    for tx_in in tx["vin"]:
        types.append(tx_in["prevout"]["scriptpubkey_type"])
    return types

def get_sending_types(tx):
    types = []
    for tx_out in tx["vout"]:
        types.append(tx_out["scriptpubkey_type"])
    return types

# Returns positive index for change output, or -1 if no change
# Tx is meant to be fetched from mempool.space
def change_index(tx):
    vout = tx['vout']
    if len(vout) <= 1:
        # Single output, not way to detect change
        return -1
    input_types = get_spending_types(tx)
    output_types = get_sending_types(tx)

    # if all inputs are of the same type, and only one output of the outputs is of that type, 
    if (len(set(input_types)) == 1):
        if output_types.count(input_types[0]) == 1:
            return output_types.index(input_types[0])  
        
    # same as one of the input addresses
    prev_txouts = [tx_in["prevout"] for tx_in in tx["vin"]]
    input_script_pub_keys = [tx_out["scriptpubkey"] for tx_out in prev_txouts]
    output_script_pub_keys = [tx_out["scriptpubkey"] for tx_out in vout]
    
    shared_address = list(set(output_script_pub_keys).intersection(set(input_script_pub_keys)))

    if len(shared_address) == 1 and output_script_pub_keys.count(shared_address[0]) == 1:
        return output_script_pub_keys.index(shared_address[0])
    
    output_amounts = [int(tx_out["value"] * 100_000_000) for tx_out in vout] # stored as satoshis

    possible_index = []

    for i, amount in enumerate(output_amounts):
        if amount % 100 != 0:
            possible_index.append(i)

    if len(possible_index) == 1:
        return possible_index[0]
    
    # Result is inconclusive
    return -1
