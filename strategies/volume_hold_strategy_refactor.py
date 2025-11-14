
# 对冲策略


class SLICE:
    qty = xx
    side = 
    symbol = xx

class hedge_plan:
    qty
    ..

class AccountCredentials:


    slice_value = 50

    def excute_slice(slice):
        Order_id = client.excute_order(
            qty = slice.qty,
            side = slice.side,
            type = "Limit"
        )

        return Order_id

    def place_lighter_market_order(self, lighter_side: str, quantity: Decimal, price: Decimal):
        if not self.lighter_client:
            await self.initialize_lighter_client()

        best_bid, best_ask = self.get_lighter_best_levels()

        # Determine order parameters
        if lighter_side.lower() == 'buy':
            order_type = "CLOSE"
            is_ask = False
            price = best_ask[0] * Decimal('1.002')
        else:
            order_type = "OPEN"
            is_ask = True
            price = best_bid[0] * Decimal('0.998')


        # Reset order state
        self.lighter_order_filled = False
        self.lighter_order_price = price
        self.lighter_order_side = lighter_side
        self.lighter_order_size = quantity

        try:
            client_order_index = int(time.time() * 1000)
            # Sign the order transaction
            tx_info, error = self.lighter_client.sign_create_order(
                market_index=self.lighter_market_index,
                client_order_index=client_order_index,
                base_amount=int(quantity * self.base_amount_multiplier),
                price=int(price * self.price_multiplier),
                is_ask=is_ask,
                order_type=self.lighter_client.ORDER_TYPE_LIMIT,
                time_in_force=self.lighter_client.ORDER_TIME_IN_FORCE_GOOD_TILL_TIME,
                reduce_only=False,
                trigger_price=0,
            )
            if error is not None:
                raise Exception(f"Sign error: {error}")

            # Prepare the form data
            tx_hash = await self.lighter_client.send_tx(
                tx_type=self.lighter_client.TX_TYPE_CREATE_ORDER,
                tx_info=tx_info
            )

            self.logger.info(f"[{client_order_index}] [{order_type}] [Lighter] [OPEN]: {quantity}")

            await self.monitor_lighter_order(client_order_index)

            return tx_hash
        except Exception as e:
            self.logger.error(f"❌ Error placing Lighter order: {e}")
            return None
    def excute_hedge(hedge_plan):
        if hedge_plan.qty == 0:
            return
        # define markte method:
        place_lighter_market_order(hedge_plan)

        return res

    def wether_excute(new_slce):

        if new_slice.qty > min_order_size and new_slice.qty * ref_price > 11 USD:
            return True
        else:
            return False

    def _waitng_fill(
            waiting_tolerance
            symbol
            slice
    ):

        while True:
            sleep(0.5)
            position = self.client.get_postion(symbol)

            if slice.side == sell:
                slice_qty = -slice_qty
            if position - self.position = slice.slice_qty:
                break

            if time.time() - time_submit >= waiting_tolerance:
                break
        
        fill_amt = position - self.position
        self.position = position

        return fill_amt

    def split_amt(filled_amt):

        hedge1_amt = random(0.45,0.55) * filled_amt

        hedge1_amt = round_to_precision(hedge1_amt)

        hedge2_amt = filled_amt - hedge1_amt

        # 容忍 10% 的瞬间波动
        if hedge1_amt * ref_price < 11u or hedge2_amt * ref_price < 11u
            hedge1_amt = filled_amt
            hedge2_amt = 0

        return hedge1_amt,hedge2_amt
    
    def check_net_position(self)
        primary_amt = self.client_primary(symbol)
        hedge1_amt = self.client_hedge1(symbol)
        hedge2_amt = self.client_hedge2(symbol)


        net_postion = primary_amt - hedge1_amt - hedge2_amt

        return primary_amt,hedge1_amt,hedge2_amt,net_postion


    def accumulate_position(self):
        market_price,ref_price = self.get_price()
        remaining_amt = round_to_precision(remaining_value/market_price)
            
        while remaining_amt > 0:
            market_price,ref_price = self.get_price()

            _,_,_,net_position = self.check_net_position()

            # 扔掉的部分会造成误差，误差累计到 11u 以上的金额时处理
            if net_position * price > 11:
                self.excute_hedge(hedge1_acount, net_position)
            
            slice_qty = slice_value / ref_price
            new_slice = SLICE(
                
            )

            if not self.wether_excute():
                break

            order_id = self.excute_slice(new_slice)

            filled_amt = self._waitng_fill()

            # 低于 10u 扔掉，不 hedge 了，懒得管
            if filled_amt * price < 10:
                remaining -= filled_amt
                continue

            hedge_amt1, hedge_amt2 = self.split_amt(filled_amt)


            self.excute_hedge(hedge_account1,hedge_amt1)
            self.excute_hedge(hedge_account2,hedge_amt2)

        net_position = self.check_net_position()
            
            
        return
    

    def flatten_position(self):
        market_price,ref_price = self.get_price()
        
        primary_amt, hedge1_amt, hedge2_amt, _ = self.check_net_position()
        
        # 与 accumulate 相反，但是多一个条件。
        # 最后一笔分配的时候，split 的 amt 应该参考 hedge_amt，不应该超过。比如 primary 最后有 50，hedge1_amt 20, hedge2_amt 30。那么就不能 spilit 成 25 25

    def run(self) -> None:

        entry_summary = self._accumulate_position(primary_idx, hedger_indices, symbol_plan)
        if entry_summary <= 0:
            logger.warning("No fills recorded for %s, skipping exit leg", symbol_plan.symbol)
        else:
            self._hold_position(symbol_plan)
            self._flatten_position(primary_idx, hedger_indices, symbol_plan, entry_summary)
            self._print_cycle_summary(symbol_plan.symbol)