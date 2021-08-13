To verify the correlator pipeline:

1. Turn on the pipeline (at least the correlator part). eg

```
python pipeline.py --ibverbs --nobeamform -C 1,2,3,4,6,7,7,7 -a 0 --ip 100.100.100.100 --useetcd
```

2. Send test vectors from another server. Eg.

```
pipeline/test_transmitters/test_tx_vectors.py -i 100.100.100.100 -f pipeline/verification/test_vectors/in_7200t_192c_352s_2p_deadbeef.dat
```

3. Turn on a receiver. Eg.

```
pipeline/test_receivers/corr_full_rx.py -i 100.100.100.1
```

4. Turn on the pipeline. Eg, from ipython:

```
import lwa352_x_control, time
x = lwa352_x_control.Lwa352XControl()

# Turn on first stage accumulation
x.corr.set_acc_length(2400)
# start immediately
x.corr.set_start_time(-1);
time.sleep(0.5)

# Turn on second stage accumulation
x.corr_acc.set_acc_length(240000)
# start immediately
x.corr_acc.set_start_time(-1)

# Set an output data rate throttle. (The python receiver isn't very fast)
x.corr_output_full.set_packet_delay(100000)
# Set appropriate destination address
x.corr_output_full.set_destination('100.100.100.1', 11111)
```

5. Wait for the `corr_full_rx.py` to write an accumulation to disk

6. Check the accumulation against the golden visibilities. Eg:

```
pipeline/verification/tests/test_corr_full_rx.py \
  -i pipeline/verification/test_vectors/corr_7200t_2400a_192c_352s_2p_deadbeef.dat \
  -u pipeline/test_receivers/test_corr_full_rx_1269600t_0c_192nc_240000a.dat
```

7. Hopefully... PASS!
