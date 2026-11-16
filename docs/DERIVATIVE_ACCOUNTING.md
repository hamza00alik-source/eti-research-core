# Contract-level derivative funding primitive

`eti.derivative_accounting.calculate_contract_funding_payment` is a pure accounting primitive for one settlement event and an explicitly supplied position.

- **Linear contract:** `contract_size` is base-currency units per contract; quote notional is contracts × contract size × mark price.
- **Inverse contract:** `contract_size` is quote-currency value per contract; base notional is quote notional divided by mark price.
- Positive funding rate means a long pays and a short receives. Negative rate reverses this.
- The returned funding cost is positive for a payment and negative for a credit.
- Settlement is supported only in the instrument's base or quote currency. Other settlement currencies require an explicit FX conversion source and are rejected.

## Deliberate integration boundary

This primitive is **not yet used by the ETI backtester or research runner**. Those systems currently use normalized exposure positions in [-1, 1], not an actual contract quantity and account equity ledger. They do not yet carry a mark price at every funding timestamp, account-currency FX conversions, margin mode, collateral, liquidation, or exchange-specific inverse/linear contract settlement rules. Plugging this helper into normalized positions without a real position-sizing model would create false precision.

Before integration, ETI must define and test a position-sizing contract that maps account equity, target exposure, leverage, contract multiplier and exchange precision into a real contract quantity. The same exact quantity must be used by simulated fills, funding cashflows, margin and liquidation calculations. Funding events must also be independently sourced and complete; the existing expected-timestamp list only validates internal consistency, not source authenticity.


## Governed research gate

The budget allocator marks all swap/future candidates economically ineligible until this contract-level accounting primitive is integrated with real position sizing, mark-price history, account equity, margin and liquidation. The governed research runner independently rejects non-spot market types before fetching market data or fitting models. It also rejects short spot research until borrow and financing costs are modeled. These gates prevent the normalized-exposure backtester from presenting incomplete derivative or short economics as cost-adjusted evidence.
