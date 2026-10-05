from ai_asic.hardware.bitcoin_header import (
    extract_nonce,
    extract_slots,
    prepare_asic_job,
    validate_header,
)
from ai_asic.hashing.methods import SoftwareHashMethod
from ai_asic.hashing.network import HashNetwork
from ai_asic.hashing.neuron import HashNeuron, MiningNeuron
from ai_asic.hashing.recursive import RecursiveEngine


def test_bitcoin_header_roundtrip():
    slots = list(range(1, 13))
    header = prepare_asic_job(slots, 0x12345678, timestamp=0)
    assert len(header) == 80
    assert validate_header(header)
    assert extract_nonce(header) == 0x12345678
    assert extract_slots(header) == slots


def test_hash_neuron_deterministic_and_ranged():
    seed = bytes(range(32))
    n = HashNeuron(seed)
    a = n.forward(b"hello")
    b = n.forward(b"hello")
    assert a == b
    assert 0.0 <= a <= 1.0
    assert n.forward(b"world") != a


def test_network_forward_and_predict():
    net = HashNetwork(8, 16, 8, 4)
    out = net.forward(b"input-data")
    assert len(out) == 4
    assert all(0.0 <= v <= 1.0 for v in out)
    idx, conf = net.predict(b"input-data")
    assert 0 <= idx < 4
    # Deterministic for a fixed network + input.
    assert net.predict(b"input-data") == (idx, conf)


def test_network_serialize_roundtrip():
    net = HashNetwork(8, 16, 8, 4)
    restored = HashNetwork.deserialize(net.serialize())
    assert restored.predict(b"abc") == net.predict(b"abc")


def test_recursive_engine_software():
    net = HashNetwork(8, 16, 8, 3)
    engine = RecursiveEngine(net, passes=11, jitter=0.02, seed_rotation=True,
                             hash_method=SoftwareHashMethod())
    result = engine.infer(b"some input")
    assert result.valid_passes == 11
    assert 0 <= result.consensus.prediction < 3
    assert 0.0 <= result.consensus.confidence <= 1.0
    summary = result.statistical_summary()
    assert "class_distribution" in summary


def test_recursive_engine_no_hardware_path():
    net = HashNetwork(8, 16, 8, 3)
    engine = RecursiveEngine(net, passes=5, jitter=0.0)
    assert not engine.is_using_hardware()
    result = engine.infer(b"x")
    assert result.valid_passes == 5


def test_mining_neuron_software():
    n = MiningNeuron(input_dim=4, output_dim=4, salt=7, nonce_start=0, nonce_end=5000)
    nonce = n.forward([0.1, 0.2, 0.3, 0.4])
    assert isinstance(nonce, int)
    # Deterministic for the same input.
    assert n.forward([0.1, 0.2, 0.3, 0.4]) == nonce
