import Examples
ex = Examples.GuiEtAl2018()
tx_data, rx_data = ex.run(5) # 5 is the number of bursts
ex.evaluate_results(tx_data, rx_data)
from QualityAssessment import BitErrorRatio
ber = BitErrorRatio(ex.constellation)
ber_value, n_err, n_bits = ber.compute(tx_data["symbols"], rx_data["symbols"])
print("The bit error ratio is", ber_value)
