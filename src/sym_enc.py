def symm_encryp_create(input_message, input_secret_key):
    # Alice creates a ciphertext and a secret key from the input message
    # Bob takes the cipertext along with the secret key to decrypt and generate the original message
    # Eve takes in just the ciphertext text and tries to decrpyt it without using key
 
    Alice_out_ciphertext = model('Alice', input_message, input_secret_key)

    Bob_out_message_decrypted = model('Bob', Alice_out_ciphertext, input_secret_key)

    Eve_out_message_decrypted = model( 'Eve', Alice_out_ciphertext)
  
    return Bob_out_message_decrypted, Eve_out_message_decrypted